from transformers import PreTrainedTokenizer


class WMTProcessor:

    def __init__(
            self,
            src_lang: str = None,
            tgt_lang: str= None,
            tokenizer: PreTrainedTokenizer= None,
            gen_tokenizer: PreTrainedTokenizer= None,
            max_length: int= None,
            is_gen: bool= None,
            is_t5: bool= None,
            is_s2s: bool= None
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = tokenizer
        self.gen_tokenizer = gen_tokenizer
        self.max_length = max_length
        self.is_gen = is_gen
        self.is_s2s = is_s2s
        self.is_t5 = is_t5

    def processing(self, examples):
        if self.is_s2s:
            return self.wmt_s2s_preprocess_function(examples)
        else:
            return self.wmt_casual_preprocess_function(examples)

    def wmt_casual_preprocess_function(
            self,
            examples,
            is_gen=False
    ):

        inputs = [ex[self.src_lang] for ex in examples["translation"]]
        targets = [ex[self.tgt_lang] for ex in examples["translation"]]

        if not is_gen:
            texts = [
                f'<{self.src_lang}>{source}<{self.tgt_lang}>{self.gen_tokenizer.eos_token}{target}' for source, target
                in zip(inputs, targets)
            ]

            model_inputs = self.tokenizer(
                texts, max_length=self.max_length, truncation=True,
                return_length=True

            )

        else:
            texts = [
                f'<{self.src_lang}>{source}<{self.tgt_lang}>' for source, target in zip(inputs, targets)
            ]

            model_inputs = self.tokenizer(
                texts,
                text_target=targets,
                max_length=self.max_length,
                truncation=True,
                return_length=True
            )

        return model_inputs

    def wmt_s2s_preprocess_function(
            self,
            examples,
    ):
        if self.is_t5:
            task = f'translate from {self.src_lang} to {self.tgt_lang}'
            inputs = [task + ex[self.src_lang] for ex in examples['translation']]
        else:
            inputs = [ex[self.src_lang] for ex in examples['translation']]
        targets = [ex[self.tgt_lang] for ex in examples["translation"]]

        model_inputs = self.tokenizer(
            inputs,
            text_target=targets,
            max_length=self.max_length,
            truncation=True,
            return_length=True
        )

        return model_inputs
