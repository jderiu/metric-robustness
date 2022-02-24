import json, os
import wandb
import argparse

from transformers import BlenderbotSmallForConditionalGeneration, BlenderbotSmallTokenizer, RobertaTokenizer
from transformers.models.blenderbot.tokenization_blenderbot import BlenderbotTokenizer
from transformers.models.blenderbot.modeling_blenderbot import BlenderbotForConditionalGeneration

from src.reinforcement_learning.data_loader import DataLoader
from src.reinforcement_learning.actor_critic_loop import ActorCritic
from src.evaluate_metrics.wrappers import get_metric
from src.reinforcement_learning.utils.dialogue_system_wrapper import BlenderWrapper


if __name__ == '__main__':
    with open('config/rl_config.json', 'rt', encoding='utf-8') as ifile:
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

    eval_batch_size = config['eval_batch_size']
    checkpoint_ret_nr = config['checkpoint_ret_nr']
    max_context_len = config['max_context_len']
    max_length = config['max_len']
    n_turns = config['n_turns']
    n_steps = config['n_steps']
    device = config['device']

    gamma = config['gamma']
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    memory_sample_batch = config['memory_sample_batch']
    max_len_dial = config['max_len_dial']
    valid_steps = config['valid_steps']

    domain = args.domain if args.domain else config['domain']
    blender_bot_version = args.bot_type if args.bot_type else config['blender_bot_version']
    experiment_name_appendix = args.appendix if args.appendix else config['experiment_name']
    metric = args.metric if args.metric else config['metric']
    max_len_dial = args.max_len_dial if args.max_len_dial else config['max_len_dial']
    batch_size = args.batch_size if args.batch_size else config['batch_size']
    dial_batch_size = args.dial_batch_size if args.dial_batch_size else config['dial_batch_size']
    n_turns = args.n_turns if args.n_turns else config['n_turns']
    seed = args.seed

    checkpoint_mlm_nr = config['checkpoint_mlm_nr'][domain]
    checkpoint_ret_nr = config['checkpoint_ret_nr'][domain]

    data_loader = DataLoader(domain, seed)

    if blender_bot_version == 'facebook/blenderbot_small-90M':
        dialogue_tokenizer = BlenderbotSmallTokenizer.from_pretrained(blender_bot_version)
        dialogue_model = BlenderbotSmallForConditionalGeneration.from_pretrained(blender_bot_version)
        dialogue_tokenizer.sep_token = '    '
        bot_type = 'small'
    else:
        dialogue_tokenizer = BlenderbotTokenizer.from_pretrained(blender_bot_version)
        dialogue_model = BlenderbotForConditionalGeneration.from_pretrained(blender_bot_version)
        dialogue_tokenizer.sep_token = '    '
        bot_type = 'large'

    experiment_name = f'AC_{metric}_EP{n_turns}_D{domain}_BL{bot_type}_A{experiment_name_appendix}_L{max_len_dial}'
    tag = f'rl-experiments-final-{n_turns}'
    logging_path = os.path.join('logging', 'rl',tag, domain, metric, f'len_{max_len_dial}', bot_type, f'run_{experiment_name_appendix}')
    models_path = os.path.join('models', 'rl', tag, domain, metric, f'len_{max_len_dial}', bot_type, f'run_{experiment_name_appendix}')
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    auto_metric = get_metric(
        metric,
        domain=domain,
        max_length=max_length,
        device=device,
        checkpoint_ret_nr=checkpoint_ret_nr,
        checkpoint_mlm_nr=checkpoint_mlm_nr,
    )

    wandb.init(project=tag, name=experiment_name)
    value_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    dialogue_model.to(device)
    bl_wrapper_policy = BlenderWrapper(
        model=dialogue_model,
        tokenizer=dialogue_tokenizer,
        is_policy=True,
        max_context_len=max_context_len,
        max_len_dial=max_len_dial,
        device=device,
        name='BL-RL',
        beam_size=10,
        n_ret=1
    )

    dialogue_partners = []
    if blender_bot_version == 'facebook/blenderbot_small-90M':
        dialogue_base_model = BlenderbotSmallForConditionalGeneration.from_pretrained(blender_bot_version)
    else:
        dialogue_base_model = BlenderbotForConditionalGeneration.from_pretrained(blender_bot_version)
    dialogue_base_model.to(device)
    bl_wrapper = BlenderWrapper(
        model=dialogue_base_model,
        tokenizer=dialogue_tokenizer,
        is_policy=False,
        max_context_len=max_context_len,
        max_len_dial=max_len_dial,
        device=device,
        name='BL-Base',
        beam_size=10,
        n_ret=1
    )
    dialogue_partners.append(bl_wrapper)

    actor_critic = ActorCritic(
        policy_wrapper=bl_wrapper_policy,
            dialogue_partners=dialogue_partners,
            value_fct_location=f'roberta-base',
            reward_fct=auto_metric,
            value_tok=value_tokenizer,
            logging_path=logging_path,
            models_path=models_path,
            gamma=gamma,
            dial_batch_size=dial_batch_size,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_context_len=max_context_len,
            n_turns=n_turns,
            valid_steps=valid_steps,
            device=device
    )

    actor_critic.optimize_ac(data_loader, n_steps, eval_0=True)