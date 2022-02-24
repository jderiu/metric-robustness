import json, os
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
    context_len = 2
    all_turns = [item for sublist in dialogues for item, _ in sublist]
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    with open(f'roberta_ret_{fname}set_single.txt', 'wt', encoding='utf-8') as ofile:
        for dialogue in dialogues:
            current_context = [dialogue[0][0]]
            for turn, _ in dialogue[1:]:
                pos_example = tokenizer.sep_token.join(current_context[-context_len + 1:] + [turn])
                ofile.write(f'{pos_example}\t{1}\n')
                neg_samples_turns = random.sample(all_turns, k=1)
                for neg_sample_turn in neg_samples_turns:
                    neg_sample = tokenizer.sep_token.join(current_context[-context_len + 1:] + [neg_sample_turn])
                    ofile.write(f'{neg_sample}\t{0}\n')
                current_context.append(turn)

def load_dialogues(ifile):
    all_dialgoues = []
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
        negative_samples = line.split('\t')[2].split('|')
        current_dialogue.extend([(turn0, []), (turn1, negative_samples)])
    return all_dialgoues


for fname in fnames:
    with open(f'{fname}_none_original.txt', 'rt', encoding='utf-8') as ifile:
        all_dialogues = load_dialogues(ifile)
        create_train_ret_format(all_dialogues, fname)
        #create_mlm_format(all_dialogues, fname)
