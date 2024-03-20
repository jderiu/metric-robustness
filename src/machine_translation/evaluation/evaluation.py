from typing import List, Dict
import evaluate

metric = evaluate.load("sacrebleu")

def analyse_lexical_variety(all_outputs: List[Dict]):
    all_mt_words, all_ref_words = set(), set()
    n_mt_words, n_ref_words = 0, 0
    for odict in all_outputs:
        output_toks = odict['mt'].split()
        ref_toks = odict['ref'].split()
        all_mt_words.update(output_toks)
        all_ref_words.update(ref_toks)

        n_mt_words += len(output_toks)
        n_ref_words += len(ref_toks)

    return len(all_mt_words)/n_mt_words, len(all_ref_words)/n_ref_words


def compute_bleu(all_outputs: List[Dict]):
    all_refs, all_mt = [], []
    for odict in all_outputs:
        mt = odict['mt']
        ref = odict['ref']
        all_refs.append([ref])
        all_mt.append(mt)

    result = metric.compute(predictions=all_mt, references=all_refs)
    return result["score"]


def compute_avg_rating(all_outputs: List[Dict]):
    return sum([odict['rating'] for odict in all_outputs])/len(all_outputs)