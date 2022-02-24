from src.reinforcement_learning.utils.dialogue_system_wrapper import DialogueSystemWrapper
from typing import List

def sample_bot_talk(
        batch_seed_context: List[List[str]],
        model_wrapper: DialogueSystemWrapper,
        partner_model_wrapper: DialogueSystemWrapper,
        n_turns: int,
        alternative_model_wrapper: DialogueSystemWrapper = None,
        evaluate: bool = False
):
    model_wrapper.model.eval()
    partner_model_wrapper.model.eval()
    batch_size = len(batch_seed_context)
    all_outputs = [[] for _ in range(batch_size)]

    models = [
        model_wrapper, partner_model_wrapper
    ]

    #random.shuffle(models)
    for idx in range(n_turns):
        current_model = models[idx % 2]
        reply_strs = current_model.generate_batch_responses(batch_seed_context, explore=not evaluate)
        alt_reply_str = None
        if alternative_model_wrapper is not None:
            alt_reply_str = alternative_model_wrapper.generate_batch_responses(batch_seed_context, explore=False)
        for didx, context in enumerate(all_outputs):
            context.append({
                'context': [x for x in batch_seed_context[didx]],
                'response': reply_strs[didx],
                'is_tracked': current_model.is_policy,
                'name': current_model.name,
                'alternative': alt_reply_str[didx] if alt_reply_str is not None else ''
            })
            batch_seed_context[didx].append(reply_strs[didx])
    return all_outputs
