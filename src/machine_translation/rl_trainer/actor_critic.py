import os
import wandb
import torch
import random

from transformers import DataCollator
from transformers.trainer import RandomSampler, SequentialSampler
from datasets import Dataset

from src.machine_translation.metrics.default_metric import AutoMetric
from src.machine_translation.policies.default_value_funct import ValueFunction
from src.machine_translation.policies.default_policy import Policy
from src.machine_translation.data_processing.processing_functions import WMTProcessor
from src.machine_translation.evaluation.evaluation import analyse_lexical_variety, compute_bleu, compute_avg_rating
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from tqdm import tqdm

class ActorCritic:

    def __init__(
            self,
            policy: Policy,
            value_fct: ValueFunction,
            reward_fct: AutoMetric,
            gen_batch_size: int,
            batch_size: int,
            eval_batch_size: int,
            gradient_accumulation_steps: int,
            gamma: float,
            logging_path: str,
            models_path: str,
            valid_steps: int
    ):
        self.policy = policy
        self.value_fct = value_fct

        self.reward_fct = reward_fct

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
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

    def eval_policy(
            self,
            dataset:
            Dataset,
            data_processor,
            data_collator,
            ep: int
    ):
        sequential_sampler = SequentialSampler(dataset)
        batch_idx = []
        all_outputs = []
        for idx, sample_idx in tqdm(enumerate(sequential_sampler, start=1), total=len(sequential_sampler)):
            batch_idx.append(sample_idx)
            if not idx % self.eval_batch_size == 0 or idx == len(dataset):
                continue
            batch = dataset[batch_idx]
            batch_idx = []
            collated_batch = self.create_input_batch(batch, data_processor, data_collator)
            sampled_outputs = self.policy.generate_batch_responses(collated_batch, explore=False)

            merged_outputs = [{
                'src': b[data_processor.src_lang],
                'ref': b[data_processor.tgt_lang],
                'mt': sampled_outputs[idx],
            } for idx, b in enumerate(batch['translation'])]

            all_outputs.extend(merged_outputs)

        ratings = self.reward_fct.batch_rating(all_outputs)
        for odict, rating in zip(all_outputs, ratings):
            odict['rating'] = rating

        mt_lex_var, ref_lex_var = analyse_lexical_variety(all_outputs)
        bleu_score = compute_bleu(all_outputs)
        avg_reward = compute_avg_rating(all_outputs)

        wandb.log(
            {
                'eval/mt_lex_var': float(mt_lex_var),
                'eval/ref_lex_var': float(ref_lex_var),
                'eval/context_bleu_score': float(bleu_score),
                'eval/avg_reward': float(avg_reward),
            }
        )

        print(f'Epoch: {ep}\tHyp Lexical Variety: {mt_lex_var:.2f}\tRef Lexical Variety: {ref_lex_var:.2f}\tContext BLEU: {bleu_score:.2f}\n')

        with open(os.path.join(self.logging_path, f'outs_{ep}.txt'), 'wt', encoding='utf-8') as ofile:
            ofile.write(f'Avg Reward: {avg_reward:.2f}\n')
            ofile.write(
                f'Hyp Lexical Variety: {mt_lex_var:.2f}\tRef Lexical Variety: {ref_lex_var:.2f}\tContext BLEU: {bleu_score:.2f}\n')
            for odict in all_outputs:
                src = odict['src']
                ref = odict['ref']
                mt = odict['mt']
                rating = str(odict['rating'])

                ostr = '\t'.join([rating, src, mt, ref]) + '\n'
                ofile.write(ostr)

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

    def create_value_input_batch(self, batch, data_processor, data_collator,sampled_outputs=None):
        if sampled_outputs is not None:
            for i, ex in enumerate(batch["translation"]):
                ex[data_processor.tgt_lang] = sampled_outputs[i]
        processed_batch = data_processor.wmt_seq_clf_processing_function(batch)
        transposed_list = [{
            'input_ids': processed_batch.data['input_ids'][i],
            'attention_mask': processed_batch.data['attention_mask'][i],
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
        masked_stp1 = (values[:, 1:] * (labels[:, 1:].to(self.device) != -100)).detach()
        masked_st = values[:, :-1] * (labels[:, :-1].to(self.device) != -100)

        diff = self.gamma * masked_stp1 - masked_st
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
            value_data_collator: DataCollator,
            value_data_processor: WMTProcessor,
            n_episodes: int,
            advantage_type: str = 'sample',
            eval_0=False
    ):
        step = 1

        if eval_0:
            avg_reward = self.eval_policy(
                valid_dataset,
                data_processor,
                data_collator,
                0
            )

            print(avg_reward)
            wandb.log({'eval/reward': float(avg_reward)})
        for ep in range(1, n_episodes):
            random_sampler = RandomSampler(train_dataset)
            n_samples = len(random_sampler)
            print(f'Epoch: {ep}')
            batch_idx = []
            current_accumulation = self.gradient_accumulation_steps
            policy_losses, value_losses, rewards, lm_losses = [],[],[],[]
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

                if self.value_fct is None:
                    values = self.policy.compute_value(collated_batch)
                else:
                    value_batch = self.create_value_input_batch(batch, data_processor, data_collator, sampled_outputs)
                    values = self.value_fct.compute_value(value_batch).unsqueeze(-1)

                logits = self.policy(collated_batch)
                lm_loss = self.compute_lm_loss(logits, collated_batch['labels'])

                if advantage_type == 'sample':
                    advantages = self.compute_sample_advantage(values, ratings)
                    value_loss = advantages ** 2
                    policy_loss = (advantages.detach() * lm_loss.mean(1))
                else:
                    advantages = self.compute_token_advantage(
                        values.squeeze(-1),
                        collated_batch['labels'],
                        collated_batch['length'],
                        ratings
                    )
                    value_loss = (advantages ** 2).mean(1)
                    policy_loss = (advantages.detach() * lm_loss).mean(1)

                full_loss = policy_loss.sum() + value_loss.sum()

                full_loss.backward()
                current_accumulation -= 1
                policy_losses.extend(policy_loss.detach().cpu().tolist())
                lm_losses.extend(lm_loss.mean(1).detach().cpu().tolist())
                value_losses.extend(value_loss.detach().cpu().tolist())
                rewards.extend(ratings)

                if current_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0)
                    self.policy_opt.step()
                    self.policy_opt.zero_grad()
                    current_accumulation = self.gradient_accumulation_steps

                    avg_policy_loss = sum(policy_losses)/len(policy_losses)
                    avg_lm_loss = sum(lm_losses)/len(lm_losses)
                    avg_value_loss = sum(value_losses)/len(value_losses)
                    avg_rewards = sum(rewards)/len(rewards)

                    print(f'Sample: {idx}/{n_samples}\tPolicy Loss: {float(avg_policy_loss):.4f}\tValue Loss: {float(avg_value_loss):.4f}\tReward: {float(avg_rewards):.4f}\tLM Loss: {float(avg_lm_loss):.4f}')
                    wandb.log(
                        {
                            'train/policy_loss': float(avg_policy_loss),
                            'train/value_loss': float(avg_value_loss),
                            'train/avg_reward': float(avg_rewards),
                            'train/lm_loss': float(avg_value_loss)
                        }
                    )

                step += 1

            self.eval_policy(
                valid_dataset,
                data_processor,
                data_collator,
                ep
            )
