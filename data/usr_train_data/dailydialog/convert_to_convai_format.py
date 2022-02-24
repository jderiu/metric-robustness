import json
import random
from transformers import RobertaTokenizer
fnames = ['train', 'valid', 'test']


def create_mlm_format(dialogues, fname):
    context_len = 4
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    with open(f'roberta_{fname}set.txt', 'wt', encoding='utf-8') as ofile:
        for dialogue in dialogues:
            current_context = []
            for turn in dialogue:
                current_context.append(turn)
                sample = tokenizer.sep_token.join(current_context[-context_len:])
                ofile.write(f'{sample}\n')


def create_train_ret_format(dialogues, fname):
    context_len = 4
    all_turns = [item for sublist in dialogues for item in sublist]
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    with open(f'roberta_ret_{fname}set.txt', 'wt', encoding='utf-8') as ofile:
        for dialogue in dialogues:
            current_context = [dialogue[0]]
            for turn in dialogue[1:]:
                pos_example = tokenizer.sep_token.join(current_context[-context_len + 1:] + [turn])
                ofile.write(f'{pos_example}\t{1}\n')
                neg_samples_turns = random.sample(all_turns, k=1)
                for neg_sample_turn in neg_samples_turns:
                    neg_sample = tokenizer.sep_token.join(current_context[-context_len + 1:] + [neg_sample_turn])
                    ofile.write(f'{neg_sample}\t{0}\n')
                current_context.append(turn)

def create_original_format(dialogues, fname):
    all_turns = [item for sublist in dialogues for item in sublist]
    with open(f'{fname}_none_original.txt', 'wt', encoding='utf-8') as ofile:
        for dialogue in dialogues:
            current_exchange_id = 1
            current_exchange = []
            for text in dialogue:
                current_exchange.append(text)
                if len(current_exchange) == 2:
                    negative_samples = random.sample(all_turns, k=15)
                    negative_samples_str = '|'.join(negative_samples)
                    oline = f'{current_exchange_id} {current_exchange[0]}\t{current_exchange[1]}\t{negative_samples_str}\n'
                    ofile.write(oline)
                    current_exchange = []
                    current_exchange_id += 1

for fname in fnames:
    with open(f'{fname}.json', 'rt', encoding='utf-8') as ifile:
        all_dialogues = []
        for line in ifile:
            jdict = json.loads(line)
            dialogue_dict = jdict['dialogue']
            turns = []
            for turn in dialogue_dict:
                text = turn['text']
                turns.append(text)
            all_dialogues.append(turns)
        create_original_format(all_dialogues, fname)
        create_train_ret_format(all_dialogues, fname)
        create_mlm_format(all_dialogues, fname)
