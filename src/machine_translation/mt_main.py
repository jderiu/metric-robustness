import json, argparse, os, wandb
from datasets import load_dataset

from transformers import DataCollatorForSeq2Seq

from src.machine_translation.policies.default_policy import Seq2SeqPolicy
from src.machine_translation.metrics.default_metric import CometMetric
from src.machine_translation.data_processing.processing_functions import WMTProcessor

from src.machine_translation.rl_trainer.actor_critic import ActorCritic

if __name__ == '__main__':
    with open('config/mt_rl_config.json', 'rt', encoding='utf-8') as ifile:
        config = json.load(ifile)


    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--domain')
    parser.add_argument('-m', '--metric')
    parser.add_argument('-a', '--appendix')
    parser.add_argument('-b', '--bot_type')
    parser.add_argument('-l', '--max_len_dial', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--dial_batch_size', type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--n_turns', default=1, type=int)
    args = parser.parse_args()

    policy_name = config['policy']
    hf_dataset_name = config['hf_dataset_name']
    metric_name = config['metric_name']
    eval_batch_size = config['eval_batch_size']
    max_length = config['max_length']
    n_epochs = config['n_epochs']
    device = config['device']
    base_path = config['base_path']
    from_pretrained = config['from_pretrained']

    gamma = config['gamma']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    valid_steps = config['valid_steps']

    experiment_name_appendix = args.appendix if args.appendix else config['experiment_name']
    metric_name = args.metric if args.metric else config['metric_name']
    batch_size = args.batch_size if args.batch_size else config['batch_size']
    seed = args.seed

    tag = f'{hf_dataset_name["path"]}-{policy_name}-{metric_name}'

    logging_path = os.path.join('logging', 'mt-rl',tag, f'run_{experiment_name_appendix}')
    models_path = os.path.join('models', 'mt-rl', tag, f'run_{experiment_name_appendix}')
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    policy = Seq2SeqPolicy(
        policy_name, max_length, from_pretrained=from_pretrained
    )
    policy.to(device)

    processor = WMTProcessor(
        src_lang=hf_dataset_name['language_pair'][0],
        tgt_lang=hf_dataset_name['language_pair'][1],
        tokenizer=policy.tokenizer,
        max_length=max_length,
        is_s2s=True
    )

    dataset = load_dataset(**hf_dataset_name, cache_dir=f'{base_path}/hf_cache')

    train = dataset['train']
    valid = dataset['validation']

    data_collator = DataCollatorForSeq2Seq(policy.tokenizer, pad_to_multiple_of=4)

    metric = CometMetric()

    trainer = ActorCritic(
        policy=policy,
        reward_fct=metric,
        gamma=gamma,
        batch_size=batch_size,
        gen_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_path=logging_path,
        models_path=models_path,
        valid_steps=valid_steps
    )

    wandb.init(project="Metric robustness", entity="deri", name=f'test')

    trainer.optimize_ac(
        train_dataset=train,
        valid_dataset=valid,
        data_collator=data_collator,
        data_processor=processor,
        n_episodes=n_epochs,
        advantage_type='sample'
    )

