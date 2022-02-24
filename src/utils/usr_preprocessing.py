import os
import torch
import numpy as np
from transformers import RobertaTokenizer
from torch.utils.data.dataset import Dataset

from typing import Dict

class ClassificationLineByLineDataset(Dataset):
    def __init__(self, tokenizer: RobertaTokenizer, file_path: str, block_size: int, n_labels: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('\t')[0] for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        with open(file_path, encoding="utf-8") as f:
            labels = [int(line.split('\t')[1]) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        n_samples = len(labels)
        batch_labels = np.zeros(shape=(n_samples, n_labels))
        batch_labels[np.arange(n_samples), labels] = 1
        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long) , 'labels': l} for e, l in zip(self.examples, labels)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class PersonaChatMLMLoader():
    def __init__(self, tokenizer: RobertaTokenizer, domain):
        self.data_path = f'auto_judge/USR/{domain}'
        self.spe_tok = tokenizer.sep_token
        self.tokenizer = tokenizer
        self.train_set = self.load_dialogues('train_none_original.txt')
        self.valid_set = self.load_dialogues('valid_none_original.txt')
        self.test_set = self.load_dialogues('test_none_original.txt')

        self.context_len = 4

        self.train_samples = self.preprocess_data(self.train_set)
        self.valid_samples = self.preprocess_data(self.valid_set)
        self.test_samples = self.preprocess_data(self.test_set)

    def process_single_dialogue(self, dialogue):
        samples = []
        for tidx in range(len(dialogue)):
            sample = dialogue[max([0, tidx - self.context_len]): tidx + 1]
            sample_str = self.spe_tok.join(sample)
            samples.append(sample_str)
        return samples

    def preprocess_data(self, dialogues):
        all_samples = []
        for dialogue in dialogues:
            dialogue_samples = self.process_single_dialogue(dialogue)
            all_samples.extend(dialogue_samples)
        return all_samples

    def load_dialogues(self, fname):
        all_dialgoues = []
        with open(os.path.join(self.data_path, fname), 'rt', encoding='utf-8') as ifile:
            current_tid = 0
            current_dialogue = []
            for line in ifile.readlines():
                line = line.replace('\n', '')
                tid = int(line.split()[0])
                if tid < current_tid:
                    current_dialogue = []
                    all_dialgoues.append(current_dialogue)
                current_tid = tid
                turn0 = ' '.join(line.split('\t')[0].split()[1:])
                turn1 = line.split('\t')[1]
                current_dialogue.extend([turn0, turn1])
        return all_dialgoues

    def create_input_for_sample(self, context, response):
        input_text = self.tokenizer.sep_token.join(context + [response])
        encoded_text = self.tokenizer(input_text).data
        response_idx = [i for i, x in enumerate(encoded_text['input_ids']) if x == self.tokenizer.sep_token_id][-2]

        batch_input = []
        batch_labels = []
        for idx in range(response_idx + 1, len(encoded_text['input_ids']), 1):
            labels = [-100] * len(encoded_text['input_ids'])
            labels[idx] = encoded_text['input_ids'][idx]

            input = [x for x in encoded_text['input_ids']]
            input[idx] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            batch_input.append(input)
            batch_labels.append(labels)

        final_inputs = torch.cat([torch.tensor(x)[None, :] for x in batch_input], dim=0)
        final_labels = torch.cat([torch.tensor(x)[None, :] for x in batch_labels], dim=0)
        batch = {
            'input_ids': final_inputs,
            'labels': final_labels
        }
        return batch


