import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.activations import ACT2FN
from transformers import RobertaForSequenceClassification, RobertaForTokenClassification, T5ForTokenClassification


class ValueFunction(nn.Module):

    def __init__(self, *args, **kwargs):
        super(ValueFunction, self).__init__(*args, **kwargs)

    def compute_value(self, batch_input):
        pass


class SeqClassifierValueFunction(ValueFunction):

    def __init__(
            self,
            path,
            from_pretrained: bool = True,
            *args, **kwargs
    ):
        super(SeqClassifierValueFunction, self).__init__(*args, **kwargs)
        if from_pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1)
        else:
            config = AutoConfig.from_pretrained(path)
            config.problem_type = 'regression'
            self.model = AutoModelForSequenceClassification.from_config(config)

        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def compute_value(self, batch_input):
        output: SequenceClassifierOutput = self.model(
            input_ids=batch_input["input_ids"].to(self.model.device),
            attention_mask=batch_input["attention_mask"].to(self.model.device),
        )
        return output.logits.unsqueeze(-1)


class TokClassifierValueFunction(ValueFunction):

    def __init__(
            self,
            path,
            from_pretrained: bool = True,
            *args, **kwargs
    ):
        super(TokClassifierValueFunction, self).__init__(*args, **kwargs)
        if from_pretrained:
            self.model = AutoModelForTokenClassification.from_pretrained(path, num_labels=1)
        else:
            config = AutoConfig.from_pretrained(path)
            config.problem_type = 'regression'
            self.model = AutoModelForTokenClassification.from_config(config)

        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def compute_value(self, batch_input, tgt_len):
        output: SequenceClassifierOutput = self.model(
            input_ids=batch_input["input_ids"].to(self.model.device),
            attention_mask=batch_input["attention_mask"].to(self.model.device),
        )

        output_logits = torch.zeros((batch_input["input_ids"].shape[0], tgt_len), device=self.model.device)
        for i, x in enumerate(batch_input['length']):
            output_logits[i, :x] = output.logits[i, -x:, 0]

        return output_logits.unsqueeze(-1)