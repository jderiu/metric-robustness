import torch,  os, glob
from torch.nn import CrossEntropyLoss
import pickle as pkl
import numpy as np
from transformers import AutoTokenizer
from transformers.models.blenderbot.tokenization_blenderbot import BlenderbotTokenizer
from transformers.models.blenderbot.modeling_blenderbot import BlenderbotForConditionalGeneration
from transformers.models.roberta import RobertaTokenizer, RobertaForSequenceClassification, RobertaForMaskedLM
from src.metrics.adversarial_turing.interface import AdversarialTuring
from src.metrics.maude.models import TransitionPredictorMaxPool
from src.metrics.maude.utils import batch_dialogs, batchify
from pytorch_lightning.trainer.trainer_io import load_hparams_from_tags_csv

from typing import List, Dict


def get_metric(
        metric,
        domain=None,
        max_length=None,
        device=None,
        checkpoint_ret_nr=None,
        checkpoint_mlm_nr=None,
):
    if metric == 'blender':
        auto_metric = BlenderLMWrapper(max_length, device)
    elif metric == 'att':
        auto_metric = ATTWrapper(device)
    elif metric == 'usr_ret':
        auto_metric = USRRetWrapper(checkpoint_ret_nr, domain, max_length, device)
    elif metric == 'usr_mlm':
        auto_metric = URSMLMWrapper(checkpoint_mlm_nr, domain, device)
    elif metric == 'usr_full_reg':
        auto_metric = URSFullRegression(checkpoint_mlm_nr, checkpoint_ret_nr, domain, max_length, device)
    elif metric == 'maude':
        auto_metric = MaudeWrapper(device)
    else:
        raise ValueError('Please use a Valid Metric')
    return auto_metric


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class AutoMetricWrapper:
    tokenizer = None
    model = None

    def __init__(self):
        pass

    def batch_rating(self, batch_sample: List[Dict]) -> List[float]:
        pass

    def single_rating(self, sample: Dict) -> float:
        pass

class URSFullRegression(AutoMetricWrapper):
    def __init__(self, checkpoint_mlm_nr, checkpoint_ret_nr, domain, max_length, device):
        super().__init__()
        self.mlm_wrapper = URSMLMWrapper(checkpoint_mlm_nr, domain, device)
        self.ret_wrapper = USRRetWrapper(checkpoint_ret_nr, domain, max_length, device)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.regr = pkl.loads(open(os.path.join('models', 'usr_models', 'regression', 'regr.pkl'), 'rb').read())
        self.vals = [(-1.012293440649907, 0.39053753419230813),
                (-1.012293440649907, 0.39053753419230813),
                (0.8604811806918587, 0.30259338627812254),
                (0.8604811806918587, 0.30259338627812254),
                (0.8472124995904354, 0.27987823011851076)]

    def batch_rating(self, batch_sample: List[Dict]) -> List[float]:
        mlm_rewards = self.mlm_wrapper.batch_rating(batch_sample)
        ret_rewards = self.ret_wrapper.batch_rating(batch_sample)
        n_metrics = []
        metrics = [mlm_rewards, mlm_rewards, ret_rewards, ret_rewards, ret_rewards]
        for m, (mean, std) in zip(metrics, self.vals):
            m = np.array(m)
            n_metrics.append((m - mean) / std)
        pred = np.stack(n_metrics, axis=1)
        p4 = self.regr.predict(pred)/10.0
        return p4.tolist()

    def single_rating(self, sample: Dict) -> float:
        mlm_reward = self.mlm_wrapper.single_rating(sample)
        ret_reward = self.ret_wrapper.single_rating(sample)

        return (mlm_reward * ret_reward)/10.0


class URSMLMWrapper(AutoMetricWrapper):
    def __init__(self, checkpoint_mlm_nr, domain, device, model=None):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        if model is None:
            self.model = RobertaForMaskedLM.from_pretrained(
                f'models/usr_models/mlm_models/{domain}/checkpoint-{checkpoint_mlm_nr}')
        else:
            self.model = model
        self.model.to(device)
        self.device = device

    def create_input_for_sample(self, context, response):
        input_text = self.tokenizer.sep_token.join(context + [response])
        input_text = input_text[-self.tokenizer.model_max_length:]
        encoded_text = self.tokenizer(
            input_text,
        ).data
        sep_tok_indices = [i for i, x in enumerate(encoded_text['input_ids']) if x == self.tokenizer.sep_token_id]
        if len(sep_tok_indices) > 1:
            response_idx = [i for i, x in enumerate(encoded_text['input_ids']) if x == self.tokenizer.sep_token_id][-2]
        else:
            response_idx = 0

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

    def batch_rating(self, batch_sample: List[Dict]) -> List[float]:
        rewards = []
        for sample in batch_sample:
            score = self.single_rating(sample)
            rewards.append(score)
        return rewards

    def single_rating(self, sample: Dict) -> float:
        batch_size = 12
        context = sample['context']
        response = sample['response']
        input_batch = self.create_input_for_sample(context, response)
        with torch.no_grad():
            losses = []
            for bidx in range(0, input_batch['input_ids'].shape[0], batch_size):
                input_batch_slice = input_batch['input_ids'][bidx:bidx + batch_size].to(self.device)
                label_batch_slice = input_batch['labels'][bidx:bidx + batch_size].to(self.device)

                ouptput = self.model(
                    input_ids=input_batch_slice,
                    labels=label_batch_slice
                )
                loss = ouptput.loss
                losses.append((loss, input_batch_slice.shape[0]))
        loss = sum([x[0] * x[1] for x in losses]) / input_batch['input_ids'].shape[0]
        score = torch.exp(-loss)
        return float(score)


class USRRetWrapper(AutoMetricWrapper):
    def __init__(self, checkpoint_ret_nr, domain, max_length, device, model=None, tokenizer=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        else:
            self.tokenizer = tokenizer
        if model is None:
            self.model = RobertaForSequenceClassification.from_pretrained(
                f'models/usr_models/ret_models/{domain}/checkpoint-{checkpoint_ret_nr}')
        else:
            self.model = model
        self.model.to(device)
        self.max_length = max_length
        self.device = device

    def single_rating(self, sample: Dict) -> float:
        reward = self.batch_rating([sample])
        return reward[0]

    def batch_rating(self, batch_sample: List[Dict]) -> List[float]:
        input_batch = self.tokenizer(
            [self.tokenizer.sep_token.join(x['context'] + [x['response']]) for x in batch_sample],
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).data

        with torch.no_grad():
            ouptput = self.model(
                input_ids=input_batch['input_ids'].to(self.device),
                attention_mask=input_batch['attention_mask'].to(self.device)
            )
            logits = ouptput.logits
            softmax = torch.softmax(logits, dim=-1)

        ret_rewards = [float(softmax[i][1]) for i in range(len(batch_sample))]
        return ret_rewards


class MaudeWrapper(AutoMetricWrapper):

    def __init__(self, device):
        super().__init__()
        self.device = device
        experiment_path = "{}/{}/lightning_logs/version_{}".format(
            'models/maude_models', 'na_all', '20488119'
        )
        model_save_path = "{}/checkpoints/*.ckpt".format(experiment_path)
        all_saved_models = glob.glob(model_save_path)
        model_save_path = all_saved_models[0]
        tag_path = "{}/meta_tags.csv".format(experiment_path)

        self.model = TransitionPredictorMaxPool.load_from_metrics(
            weights_path=model_save_path, tags_csv=tag_path
        )
        self.model.set_hparam("corrupt_type", 'rand_utt')
        # model.hparams.corrupt_type = args.corrupt_type
        # set model on evaluation mode
        self.model.preflight_steps()
        self.model.eval()
        self.model.freeze()
        self.model.to(device)

        hparams = load_hparams_from_tags_csv(tag_path)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.fine_tune_model)

    def single_rating(self, sample: Dict) -> float:
        context = sample['context']
        response = sample['response']
        X = [self.tokenizer.tokenize(sent) for sent in context]
        X = [self.tokenizer.convert_tokens_to_ids(sent) for sent in X]
        Y = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(response)
        )

        X = [
            self.tokenizer.build_inputs_with_special_tokens(sent)
            for sent in X
        ]
        X, X_len, X_dial_len = batch_dialogs([X])
        Y = self.tokenizer.build_inputs_with_special_tokens(Y)
        Y, Y_len = batchify([Y], False)
        X_len = torch.from_numpy(X_len)
        Y_len = torch.from_numpy(Y_len)

        score = self.model(
            X.to(self.device),
            X_len,
            X_dial_len,
            Y.to(self.device),
            Y_len
        )


        return score

    def batch_rating(self, batch_sample: List[Dict]) -> List[float]:
        rewards = []
        for sample in batch_sample:
            score = self.single_rating(sample)
            rewards.append(score)
        return rewards


class ValueFunctWrapper(AutoMetricWrapper):
    def __init__(
            self,
            model,
            tokenizer,
            max_length: int,
            device: str
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.model.to(device)
        self.max_length = max_length
        self.device = device

    def single_rating(self, sample: Dict) -> float:
        reward = self.batch_rating([sample])
        return reward[0]

    def batch_rating(self, batch_sample: List[Dict]) -> List[float]:
        input_bach_str = [self.tokenizer.sep_token.join(x['context'][-self.max_length + 1:] + [x['response']]) for x in batch_sample]
        input_batch = self.tokenizer(
            input_bach_str,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).data

        with torch.no_grad():
            ouptput = self.model(
                input_ids=input_batch['input_ids'].to(self.device),
                attention_mask=input_batch['attention_mask'].to(self.device)
            )
            logits = ouptput.logits

        ret_rewards = [float(logits[i]) for i in range(len(batch_sample))]
        return ret_rewards


class ATTWrapper(AutoMetricWrapper):

    def __init__(self, device):
        super().__init__()
        self.model = AdversarialTuring(device == 'cpu')
        self.tokenizer = self.model.scorer.tokenizer
        self.tokenizer.model_max_length = self.model.scorer.opt.max_l_cxt
        self.tokenizer.sep_token = ' _EOS_ '

    def batch_rating(self, batch_sample: List[Dict]) -> List[float]:
        rewards = []
        for sample in batch_sample:
            score = self.single_rating(sample)
            rewards.append(float(score))
        return rewards

    def single_rating(self, sample: Dict) -> float:
        context = sample['context']
        context_str = self.tokenizer.sep_token.join(context)
        response_hyp = sample['response']

        return self.model.get_score(context_str, response_hyp)


class BlenderLMWrapper(AutoMetricWrapper):

    def __init__(self, max_length, device):
        super().__init__()
        mname = 'facebook/blenderbot-400M-distill'
        self.tokenizer = BlenderbotTokenizer.from_pretrained(mname)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(mname)
        self.model.to(device)
        self.device = device
        self.max_length = max_length

    def single_rating(self, sample: Dict) -> float:
        in_text_context = self.tokenizer.sep_token.join(sample['context'])

        input_sample = self.tokenizer(
            in_text_context,
            truncation=True,
            return_tensors='pt',
            max_length=self.tokenizer.model_max_length).data
        input_sample['labels'] = self.tokenizer(
            sample['response'],
            truncation=True,
            return_tensors='pt',
            max_length=self.tokenizer.model_max_length).data['input_ids']

        ouptput = self.model(
            input_ids=input_sample['input_ids'].to(self.device),
            attention_mask=input_sample['attention_mask'].to(self.device),
            labels=input_sample['labels'].to(self.device)
        )
        loss = float(torch.exp(-ouptput.loss))

        return float(loss)

    def batch_rating(self, batch_sample: List[Dict]) -> List[float]:
        eval_batch_size = len(batch_sample)

        input_batch = self.tokenizer.pad(
            self.tokenizer(
                [self.tokenizer.sep_token.join(x['context'][-2:]) for x in batch_sample],
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).data,
            padding=True,
            return_tensors='pt',
        )
        labels = self.tokenizer.pad(
            self.tokenizer(
                [x['response'] for x in batch_sample],
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).data,
            padding=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt'
        ).to(self.device)
        input_batch['decoder_input_ids'] = shift_tokens_right(
            labels['input_ids'], self.model.config.pad_token_id, self.model.config.decoder_start_token_id
        )
        input_batch['decoder_attention_mask'] = labels.data['attention_mask']

        with torch.no_grad():
            output = self.model(**input_batch.to(self.device))  # fix to compute loss
            loss_fct = CrossEntropyLoss(reduction='none')
            masked_lm_loss = loss_fct(output.logits.view(-1, self.tokenizer.vocab_size), labels['input_ids'].view(-1))
            avg_per_sample_mlm_loss = torch.mean(masked_lm_loss.view(eval_batch_size, -1) * input_batch['decoder_attention_mask'], dim=-1)
            scores = torch.exp(-avg_per_sample_mlm_loss)
        ret_list = [float(x) for x in scores]
        return ret_list