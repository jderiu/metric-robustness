from transformers.models.roberta import RobertaForMaskedLM
from transformers import RobertaTokenizer, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.data.datasets import LineByLineTextDataset
from transformers.trainer_callback import TrainerCallback

from src.utils.data_preprocessing import AutoMetricData
from src.evaluate_metrics.wrappers import URSMLMWrapper
from src.evaluate_metrics.eval_auto_metric import corr_to_humans
import os, json, wandb

class LoggingCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state, control, **kwargs):
        epoch = state.epoch
        eval_wrapper = URSMLMWrapper(checkpoint_mlm_nr=0, domain=domain, device=device, model=kwargs['model'])
        pearson, spearman, ratings = corr_to_humans(eval_wrapper, am_data, eval_batch_size=eval_batch_size, domain=domain, metric_name='USR MLM')

        with open(os.path.join(logging_path, f'output_{epoch}.txt'), 'wt', encoding='utf-8') as lfile:
            print(pearson)
            print(spearman)
            wandb.log({
                'eval/pearson': pearson[0],
                'eval/spearman': spearman[0]
            })
            lfile.write(f'Pearson: {pearson[0]}\tSpearman: {spearman[0]}\n')
            for x in ratings:
                oline = f'{x[0]}\t{x[1]}\t{x[2]}\t{x[3]}\t{x[4]}\n'
                lfile.write(oline)


if __name__ == '__main__':
    with open('config/usr_train_config.json', 'rt', encoding='utf-8') as ifile:
        config = json.load(ifile)

    n_steps = config['n_steps']
    device = config['device']
    batch_size = config['batch_size']
    eval_batch_size = config['eval_batch_size']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    learning_rate = config['learning_rate']
    validation_step = config['validation_step']
    save_step = config['save_step']
    logging_step = config['logging_step']
    load_checkpoint = config['load_checkpoint']
    max_length = config['max_len']
    domain = config['domain']

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForMaskedLM.from_pretrained('roberta-base')

    am_data = AutoMetricData(domain)

    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=f"data/usr_train_data/{domain}/roberta_trainset.txt",
        block_size=max_length,
    )

    valid_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=f"data/usr_train_data/{domain}/roberta_validset.txt",
        block_size=max_length,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    logging_path = os.path.join(f"logging/usr_train_logging",'mlm', domain)
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)

    training_args = TrainingArguments(
        output_dir=f"models/usr_models/mlm_models/{domain}",
        overwrite_output_dir=True,
        num_train_epochs=n_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_steps=save_step,
        save_total_limit=2,
        prediction_loss_only=True,
        eval_steps=validation_step,
        evaluation_strategy="steps",
        warmup_steps=500,
        learning_rate=learning_rate,
        ignore_data_skip=True,
        logging_steps=logging_step
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[LoggingCallback()],
    )

    trainer.train()


