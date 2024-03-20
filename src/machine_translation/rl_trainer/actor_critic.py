import os
import wandb
import torch
import random

from transformers import DataCollator
from transformers.trainer import RandomSampler
from datasets import Dataset

from src.machine_translation.metrics.default_metric import AutoMetric
from src.machine_translation.policies.default_policy import Policy
from src.machine_translation.data_processing.processing_functions import WMTProcessor

from typing import List, Dict, Callable
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

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
            policy: Policy,
            reward_fct: AutoMetric,
            gen_batch_size: int,
            batch_size: int,
            gradient_accumulation_steps: int,
            gamma: float,
            logging_path: str,
            models_path: str,
            valid_steps: int
    ):
        self.policy = policy

        self.reward_fct = reward_fct

        self.batch_size = batch_size
        self.gen_batch_size = gen_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gamma = gamma
        self.device = policy.model.device
        self.valid_steps = valid_steps

        self.policy_opt = Adam(self.policy.model.parameters(), lr=3e-06)
        self.value_opt = Adam(self.policy.model.parameters(), lr=3e-05)

        self.logging_path = logging_path
        self.models_path = models_path


    def batch_samples(self, samples, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        for idx in range(0, len(samples), batch_size):
            yield samples[idx:idx + batch_size]


    def eval_policy(self, data_loader: Dataset, ep: int):
        full_rated_convos = []
        n_iter = len(data_loader.test_seeds) / self.dial_batch_size
        for sidx, seed_batch in tqdm(
                enumerate(self.batch_samples(data_loader.test_seeds, batch_size=self.dial_batch_size)), total=n_iter,
                desc='Eval Reward'):
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
            ofile.write(
                f'Response Freq: {resp_freq:.2f}\tLexical Variety: {lexical_variety:.2f}\tContext BLEU: {bleu_score:.2f}\tContext Jaccard: {avg_jaccard:.2f}\n')
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

    def create_input_batch(self, batch, data_processor, data_collator):
        processed_batch = data_processor.wmt_s2s_preprocess_function(batch)
        transposed_list = [{
            'input_ids': processed_batch.data['input_ids'][i],
            'attention_mask': processed_batch.data['attention_mask'][i],
            'length': processed_batch.data['length'][i],
            'labels': processed_batch.data['labels'][i]
        } for i in range(len(processed_batch.encodings))]

        collated_batch = data_collator(transposed_list)
        return collated_batch

    def update_batch(self, batch, sampled_outputs, data_processor, data_collator):
        for i, ex in enumerate(batch["translation"]):
            ex[data_processor.tgt_lang] = sampled_outputs[i]
        return self.create_input_batch(batch, data_processor, data_collator)

    def random_batch_samples(self, samples, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        sample_ids = random.sample(range(len(samples)), k=batch_size)
        batch = [samples[sample_id] for sample_id in sample_ids]
        return batch

    def compute_token_advantage(self, values, labels, lengths, rewards):
        masked_stp1 = values[:, 1:]*(labels[:, 1:].to(self.device) != -100)
        masked_st = values[:, :-1]*(labels[:, :-1].to(self.device) != -100)

        diff = self.gamma*masked_stp1 - masked_st
        advantages = torch.tensor(rewards, device=self.device) - values[:, -1]

        advantages = torch.cat([diff, advantages[:, None]], dim=1)

        return advantages

    def compute_sample_advantage(self, values, rewards):
        value = torch.mean(values.squeeze(-1), dim=1)
        advantage = torch.tensor(rewards, device=self.device) - value

        return advantage


    def compute_lm_loss(self, lm_logits, labels):
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        # move labels to correct device to enable PP
        labels = labels.to(lm_logits.device)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        return loss.view(labels.shape[0], -1)

    def optimize_ac(
            self,
            train_dataset: Dataset,
            valid_dataset: Dataset,
            data_collator: DataCollator,
            data_processor: WMTProcessor,
            n_episodes: int,
            advantage_type: str = 'sample',
            eval_0=False
    ):
        step = 1

        if eval_0:
            avg_reward = self.eval_policy(valid_dataset, 0)
            print(avg_reward)
            wandb.log({'eval/reward': float(avg_reward)})
        for ep in range(1, n_episodes):
            random_sampler = RandomSampler(train_dataset)
            print(f'Epoch: {ep}')
            batch_idx = []
            current_accumulation = self.gradient_accumulation_steps
            for idx, sample_idx in enumerate(random_sampler, start=1):
                batch_idx.append(sample_idx)
                if not idx % self.batch_size == 0 or idx == len(train_dataset):
                    continue
                batch = train_dataset[batch_idx]
                batch_idx = []
                collated_batch = self.create_input_batch(batch, data_processor, data_collator)
                sampled_outputs = self.policy.generate_batch_responses(collated_batch, explore=True)

                merged_outputs = [{
                    'src': b[data_processor.src_lang],
                    'ref': b[data_processor.tgt_lang],
                    'mt': sampled_outputs[idx],
                } for idx, b in enumerate(batch['translation'])]

                ratings = self.reward_fct.batch_rating(merged_outputs)
                collated_batch = self.update_batch(batch, sampled_outputs, data_processor, data_collator)

                values = self.policy.compute_value(collated_batch)

                logits = self.policy(collated_batch)
                lm_loss = self.compute_lm_loss(logits, collated_batch['labels'])

                if advantage_type == 'sample':
                    advantages = self.compute_sample_advantage(values, ratings)
                    value_loss = (advantages**2).sum()
                    policy_loss = (advantages.detach()*lm_loss.mean(1)).sum()
                else:
                    advantages = self.compute_token_advantage(
                        values.squeeze(-1),
                        collated_batch['labels'],
                        collated_batch['length'],
                        ratings
                    )
                    value_loss = (advantages**2).mean(1).sum()
                    policy_loss = (advantages.detach() * lm_loss).mean(1).sum()

                full_loss = policy_loss + value_loss

                full_loss.backward()
                current_accumulation -= 1
                if current_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0)
                    self.policy_opt.step()
                    self.policy_opt.zero_grad()
                    current_accumulation = self.gradient_accumulation_steps


                lm_loss = lm_loss.mean()
                avg_reward = torch.tensor(ratings).mean()
                print(
                    f'Sample: {idx}\tPolicy Loss: {float(policy_loss):.4f}\tValue Loss: {float(value_loss):.4f}\tReward: {float(avg_reward):.4f}\tLM Loss: {float(lm_loss):.4f}')
                wandb.log(
                    {
                        'train/policy_loss': float(policy_loss),
                        'train/value_loss': float(value_loss),
                        'train/avg_reward': float(avg_reward),
                        'train/lm_loss': float(lm_loss)
                    }
                )

                step += 1

