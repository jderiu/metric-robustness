import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import T5ForConditionalGeneration
from transformers.activations import ACT2FN

class Policy(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Policy, self).__init__(*args, **kwargs)

    def generate_batch_responses(self, batch_input, explore: bool):
        pass


class Seq2SeqPolicy(Policy):

    def __init__(
            self,
            path: str,
            max_new_tokens: int,
            from_pretrained: bool = True,
            *args, **kwargs
    ):
        super(Seq2SeqPolicy, self).__init__(*args, **kwargs)
        if from_pretrained:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        else:
            config = AutoConfig.from_pretrained(path)
            self.model = AutoModelForSeq2SeqLM.from_config(config)

        self.li1 = nn.Linear(self.model.config.d_model, self.model.config.d_model, bias=False)
        self.li2 = nn.Linear(self.model.config.d_model, self.model.config.d_model, bias=False)
        self.value_head = nn.Linear(self.model.config.d_model, 1, bias=False)

        self.act = ACT2FN['gelu_new']
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.max_new_tokens = max_new_tokens

    def compute_value(
            self,
            batch_input,
    ):
        output: Seq2SeqLMOutput = self.model(
            input_ids=batch_input["input_ids"].to(self.model.device),
            labels=batch_input["labels"].to(self.model.device),
            attention_mask=batch_input["attention_mask"].to(self.model.device),
            output_hidden_states=True
        )
        decoder_hidden_states = output.decoder_hidden_states[-1]
        hidden_states = self.act(self.li1(decoder_hidden_states))
        hidden_states = self.act(self.li2(hidden_states))
        values = self.value_head(hidden_states)

        return values

    def forward(self, batch_input):
        return self.model(
            input_ids=batch_input["input_ids"].to(self.model.device),
            labels=batch_input["labels"].to(self.model.device),
            attention_mask=batch_input["attention_mask"].to(self.model.device)
        ).logits

    def generate_batch_responses(
            self,
            batch_input,
            explore: bool
    ):
        with torch.no_grad():
            if explore:
                output = self.model.generate(
                    input_ids=batch_input["input_ids"].to(self.model.device),
                    attention_mask=batch_input["attention_mask"].to(self.model.device),
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    forced_eos_token_id=[self.tokenizer.eos_token_id],
                    do_sample=True,
                    num_beams=3
                )
            else:
                output = self.model.generate(
                    input_ids=batch_input["input_ids"].to(self.model.device),
                    attention_mask=batch_input["attention_mask"].to(self.model.device),
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    forced_eos_token_id=[self.tokenizer.eos_token_id]
                )

            decoded_preds = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            return decoded_preds
