from transformers.models.blenderbot import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers.models.gpt2 import GPT2Tokenizer, GPT2LMHeadModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from typing import List


class DialogueSystemWrapper:
    is_policy: bool = False
    name: str = 'BaseDS'
    model: PreTrainedModel = None
    tokenizer: PreTrainedTokenizer = None
    n_ret: int= 1
    beam_size: int= 10

    def generate_batch_responses(self, batch_input, explore: bool):
        pass


class BlenderWrapper(DialogueSystemWrapper):

    def __init__(
            self,
            model: BlenderbotForConditionalGeneration,
            tokenizer: BlenderbotTokenizer,
            is_policy: bool,
            beam_size: int,
            n_ret: int,
            max_context_len: int,
            max_len_dial: int,
            device: str,
            name: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.is_policy = is_policy
        self.max_context_len = max_context_len
        self.max_len_dial = max_len_dial
        self.name = name
        self.device = device
        self.beam_size = beam_size
        self.n_ret = n_ret

    def generate_batch_responses(self, batch_input: List[str], explore: bool = False):
        str_in_contexts = [self.tokenizer.sep_token.join(context[-self.max_context_len + 1:]) for context in batch_input]
        preprocessed_contexts = self.tokenizer(str_in_contexts, truncation=True)
        padded_contexts = self.tokenizer.pad(
            preprocessed_contexts,
            return_tensors='pt'
        ).to(self.device)

        if not self.is_policy or not explore:
            #sample next turn deterministically
            reply_ids = self.model.generate(
                **padded_contexts,
                max_length=self.max_len_dial,
                num_beams=self.beam_size,
                no_repeat_ngram_size=3,
                num_return_sequences=self.n_ret
            )
        else:
            reply_ids = self.model.generate(
                **padded_contexts,
                max_length=self.max_len_dial,
                do_sample=True,
                num_beams=self.beam_size,
                top_k=0,
                #top_p=0.92,
                no_repeat_ngram_size=4,
                num_return_sequences=self.n_ret
            )

        reply_strs = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)
        return reply_strs


class DialogGPTWrapper(DialogueSystemWrapper):

    def __init__(
            self,
            model: GPT2LMHeadModel,
            tokenizer: GPT2Tokenizer,
            is_policy: bool,
            max_context_len: int,
            max_len_dial: int,
            device: str,
            name: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.is_policy = is_policy
        self.max_context_len = max_context_len
        self.max_len_dial = max_len_dial
        self.name = name
        self.device = device

    def generate_batch_responses(self, batch_input: List[List[str]], explore: bool = False):
        str_in_contexts = [self.tokenizer.sep_token.join(context[-self.max_context_len + 1:]) + self.tokenizer.eos_token for context in batch_input]
        preprocessed_contexts = self.tokenizer(str_in_contexts, truncation=True)
        padded_contexts = self.tokenizer.pad(
            preprocessed_contexts,
            return_tensors='pt'
        ).to(self.device)

        max_len = max([len(x) for x in preprocessed_contexts['input_ids']]) + 25
        if not self.is_policy or not explore:
            #sample next turn deterministically
            reply_ids = self.model.generate(
                **padded_contexts,
                max_length=max_len,
                num_beams=10,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id
            )
        else:
            reply_ids = self.model.generate(
                **padded_contexts,
                max_length=max_len,
                do_sample=True,
                num_beams=10,
                top_k=40,
                top_p=0.92,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id
            )

        reply_strs = []
        for sidx in range(len(preprocessed_contexts['input_ids'])):
            context_len = len(preprocessed_contexts['input_ids'][sidx])
            reply_str = self.tokenizer.decode(reply_ids[sidx][context_len:], skip_special_tokens=True)
            reply_strs.append(reply_str)

        return reply_strs