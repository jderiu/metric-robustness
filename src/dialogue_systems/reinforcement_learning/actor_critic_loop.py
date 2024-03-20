import os
import torch
import wandb
import random
from src.dialogue_systems.reinforcement_learning.utils.dialogue_system_wrapper import DialogueSystemWrapper
from src.dialogue_systems.evaluate_metrics.wrappers import ValueFunctWrapper, AutoMetricWrapper
from transformers.models.roberta import RobertaForSequenceClassification, RobertaTokenizer
from typing import List, Dict
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from src.dialogue_systems.reinforcement_learning.self_talk import sample_bot_talk
from src.dialogue_systems.reinforcement_learning.rate_turns import rate_convos, compute_values, compute_advantages, get_avg_reward
from src.dialogue_systems.reinforcement_learning.data_loader import DataLoader
from src.dialogue_systems.reinforcement_learning.utils.analyse_outputs import analyse_context_overlap, analyse_lexical_variety,analyse_response_variety

from tqdm import tqdm


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class ActorCritic:

    def __init__(
            self,
            policy_wrapper: DialogueSystemWrapper,
            dialogue_partners: List[DialogueSystemWrapper],
            reward_fct: AutoMetricWrapper,
            value_fct_location: str,
            value_tok: RobertaTokenizer,
            dial_batch_size: int,
            batch_size: int,
            gradient_accumulation_steps: int,
            gamma: float,
            n_turns: int,
            max_context_len: int,
            logging_path: str,
            models_path: str,
            valid_steps: int,
            device: str,
    ):
        self.policy_wrapper = policy_wrapper
        self.dialogue_partners = dialogue_partners

        self.value_fct = RobertaForSequenceClassification.from_pretrained(value_fct_location, num_labels=1)
        self.value_fct.to(device)

        self.reward_fct = reward_fct
        self.value_tok = value_tok
        self.batch_size = batch_size
        self.dial_batch_size = dial_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gamma = gamma
        self.device = device
        self.n_turns = n_turns
        self.max_context_len = max_context_len
        self.valid_steps = valid_steps

        self.policy_opt = Adam(self.policy_wrapper.model.parameters(), lr=3e-06)
        self.value_opt = Adam(self.value_fct.parameters(), lr=3e-05)
        self.critic1_eval_wrapper = ValueFunctWrapper(
            self.value_fct,
            self.value_tok,
            self.max_context_len,
            self.device,
        )

        self.logging_path = logging_path
        self.models_path = models_path

    def reset_value_fct(self, value_fct_location):
        self.value_fct = RobertaForSequenceClassification.from_pretrained(value_fct_location, num_labels=1)
        self.value_fct.to(self.device)
        self.critic1_eval_wrapper = ValueFunctWrapper(
            self.value_fct,
            self.value_tok,
            self.max_context_len,
            self.device,
        )

    def batch_samples(self, samples, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        for idx in range(0, len(samples), batch_size):
            yield samples[idx:idx + batch_size]

    def update_value_function(self, dialogue_batch: List[List[Dict]]):
        all_input = []
        losses = []
        self.value_fct.train()
        for dialogue in dialogue_batch:
            for t, turn in enumerate(dialogue):
                context = turn['context']
                context_str = self.value_tok.sep_token.join(context[-self.max_context_len+1:])
                label = turn['reward']
                if not turn['done']:
                    label += self.gamma*dialogue[t + 1]['value']

                all_input.append((context_str, label))

        back_prop_loss = 0.0
        for in_batch in self.batch_samples(all_input):
            context_batch = [x[0] for x in in_batch]
            label_batch = [x[1] for x in in_batch]
            s_t = self.value_tok.pad(self.value_tok(context_batch, truncation=True), return_tensors='pt')
            s_t['labels'] = torch.tensor(label_batch)
            outputs = self.value_fct(**s_t.to(self.device))
            back_prop_loss += float(outputs.loss)
            outputs.loss.backward()
            losses.append((float(outputs.loss), len(context_batch)))
        return back_prop_loss

    def update_policy(self, dialogue_batch: List[List[Dict]]):
        all_input = []
        tokenizer = self.policy_wrapper.tokenizer
        policy = self.policy_wrapper.model
        for dialogue in dialogue_batch:
            for turn in dialogue:
                context = turn['context']
                context_str = tokenizer.sep_token.join(context[-self.max_context_len+1:])
                response_str = turn['response']
                advantage = turn['advantage']
                all_input.append((context_str, response_str, advantage))

        batch_actor_critic_loss = 0.0
        batch_lm_loss = 0.0
        policy.model.train()
        for in_batch in self.batch_samples(all_input):
            context_batch = [x[0] for x in in_batch]
            label_batch = [x[1] for x in in_batch]
            advantage_batch = [x[2] for x in in_batch]
            s_t = tokenizer.pad(tokenizer(context_batch, truncation=True), return_tensors='pt')
            labels = tokenizer.pad(tokenizer(label_batch, truncation=True), return_tensors='pt').to(self.device)
            s_t['decoder_input_ids'] = shift_tokens_right(
                labels.data['input_ids'], policy.config.pad_token_id, policy.config.decoder_start_token_id
            )
            s_t['decoder_attention_mask'] = labels.data['attention_mask']
            advantages = torch.tensor(advantage_batch).to(self.device)
            output = policy(**s_t.to(self.device))
            loss_fct = CrossEntropyLoss(reduction='none')

            masked_lm_loss = loss_fct(output.logits.view(-1, policy.config.vocab_size), labels.data['input_ids'].view(-1))
            batch_lm_loss += float(torch.mean(masked_lm_loss))
            avg_per_sample_mlm_loss = torch.sum(masked_lm_loss.view(len(in_batch), -1)*labels.data['attention_mask'], dim=-1)

            loss = torch.sum(avg_per_sample_mlm_loss*advantages)
            loss.backward()
            batch_actor_critic_loss += float(loss)
        return batch_actor_critic_loss, batch_lm_loss

    def eval_policy(self, data_loader: DataLoader, ep: int):
        full_rated_convos = []
        n_iter = len(data_loader.test_seeds)/self.dial_batch_size
        for sidx, seed_batch in tqdm(enumerate(self.batch_samples(data_loader.test_seeds, batch_size=self.dial_batch_size)), total=n_iter, desc='Eval Reward'):
            copy_batch = [x.copy() for x in seed_batch.copy()]
            partner_model_wrapper = random.choice(self.dialogue_partners)
            sampled_dialogues = sample_bot_talk(
                batch_seed_context=copy_batch,
                model_wrapper=self.policy_wrapper,
                partner_model_wrapper=partner_model_wrapper,
                alternative_model_wrapper=None,
                n_turns=self.n_turns,
                evaluate=True
            )
            rated_covos = rate_convos(
                sampled_dialogues,
                self.reward_fct,
                self.max_context_len
            )
            full_rated_convos.extend(rated_covos)
        avg_reward = get_avg_reward(full_rated_convos)
        resp_freq = analyse_response_variety(full_rated_convos)
        lexical_variety = analyse_lexical_variety(full_rated_convos)
        bleu_score, avg_jaccard = analyse_context_overlap(full_rated_convos)

        wandb.log(
            {
                'eval/response_freq': float(resp_freq),
                'eval/lexical_variety': float(lexical_variety),
                'eval/context_bleu_score': float(bleu_score),
                'eval/context_avg_jaccard': float(avg_jaccard),
            }
        )
        with open(os.path.join(self.logging_path, f'outs_{ep}.txt'), 'wt', encoding='utf-8') as ofile:
            ofile.write(f'Avg Reward: {avg_reward:.2f}\n')
            ofile.write(f'Response Freq: {resp_freq:.2f}\tLexical Variety: {lexical_variety:.2f}\tContext BLEU: {bleu_score:.2f}\tContext Jaccard: {avg_jaccard:.2f}\n')
            for rated_convo in full_rated_convos:
                start_context = '\t'.join(rated_convo[0]['context'][:-1])
                convo_retrun = 0.0
                for turn in rated_convo:
                    context = turn['context'][-1]
                    ostr = f'(E)-{context}'
                    start_context = '\t'.join([start_context, ostr])

                    response = turn['response']
                    reward = turn['reward']
                    convo_retrun += reward
                    ostr = f'({reward:.4f})-{response}'
                    start_context = '\t'.join([start_context, ostr])
                ofile.write(f'({convo_retrun:.2f}) {start_context}\n')
        return avg_reward

    def random_batch_samples(self, samples, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        sample_ids = random.sample(range(len(samples)), k=batch_size)
        batch = [samples[sample_id] for sample_id in sample_ids]
        return batch

    def optimize_ac(self, data_loader: DataLoader, n_episodes: int, eval_0=True):
        step = 1
        if eval_0:
            avg_reward = self.eval_policy(data_loader, 0)
            print(avg_reward)
            wandb.log({'eval/reward': float(avg_reward)})
        for ep in range(1, n_episodes):
            print(f'Epoch: {ep}')
            random.shuffle(data_loader.valid_seeds)
            for sidx in range(self.valid_steps):
                seed_batch = self.random_batch_samples(data_loader.valid_seeds, batch_size=self.dial_batch_size)
                copy_batch = [x.copy() for x in seed_batch.copy()]
                partner_model_wrapper = random.choice(self.dialogue_partners)
                sampled_dialogues = sample_bot_talk(
                    batch_seed_context=copy_batch,
                    model_wrapper=self.policy_wrapper,
                    partner_model_wrapper=partner_model_wrapper,
                    alternative_model_wrapper=None,
                    n_turns=self.n_turns,
                    evaluate=False
                )
                rated_covos = rate_convos(
                    sampled_dialogues,
                    self.reward_fct,
                    self.max_context_len
                )
                avg_reward = get_avg_reward(rated_covos)

                compute_values(rated_covos, self.critic1_eval_wrapper)
                compute_advantages(rated_covos, self.gamma)
                policy_loss, lm_loss = self.update_policy(rated_covos)
                torch.nn.utils.clip_grad_norm_(self.policy_wrapper.model.parameters(), 1.0)
                self.policy_opt.step()
                self.policy_opt.zero_grad()
                value_loss = self.update_value_function(rated_covos)
                torch.nn.utils.clip_grad_norm_(self.value_fct.parameters(), 40.0)
                self.value_opt.step()
                self.value_opt.zero_grad()

                print(
                    f'Sample: {sidx}\tPolicy Loss: {float(policy_loss):.4f}\tValue Loss: {float(value_loss):.4f}\tReward: {float(avg_reward):.4f}\tLM Loss: {float(lm_loss):.4f}')
                wandb.log(
                    {
                        'train/policy_loss': float(policy_loss),
                        'train/value_loss': float(value_loss),
                        'train/avg_reward': float(avg_reward),
                        'train/lm_loss': float(lm_loss)
                    }
                )

                step += 1

            avg_reward = self.eval_policy(data_loader, ep)
            print(avg_reward)
            wandb.log({'eval/reward': float(avg_reward)})
            self.policy_wrapper.model.save_pretrained(os.path.join(self.models_path, f'policy_{ep}'))
            self.value_fct.save_pretrained(os.path.join(self.models_path, f'critic1_{ep}'))
