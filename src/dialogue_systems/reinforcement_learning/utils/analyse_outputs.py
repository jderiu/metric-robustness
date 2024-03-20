from collections import Counter
from nltk.translate.bleu_score import corpus_bleu
from typing import List, Dict


def analyse_response_variety(all_convos: List[List[Dict]]):
    response_counter = Counter()
    for convo in all_convos:
        for turn in convo:
            response_counter.update([turn['response']])
    n_max = response_counter.most_common(1)[0][1]
    n_resp = sum(response_counter.values())
    return n_max/n_resp #frequency of most common response


def analyse_lexical_variety(all_convos: List[List[Dict]]):
    all_words = set()
    n_words = 0
    for convo in all_convos:
        for turn in convo:
            resp_words = turn['response'].split(' ')
            n_words += len(resp_words)
            all_words.update(resp_words)
    return len(all_words)/n_words #ratio of different words over all words


def analyse_context_overlap(all_convos: List[List[Dict]]):
    ref = []
    hyp = []
    word_overlaps = []
    for convo in all_convos:
        for turn in convo:
            resp_words = turn['response'].split(' ')
            hyp.append(resp_words)
            context_words = ' '.join(turn['context']).split(' ')
            ref.append([context_words])
            jaccard = len(set(resp_words).intersection(context_words))/len(set(resp_words).union(context_words))
            word_overlaps.append(jaccard)
    bleu_score = corpus_bleu(ref, hyp)
    avg_jaccard = sum(word_overlaps)/len(word_overlaps)
    return bleu_score, avg_jaccard