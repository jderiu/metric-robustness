from src.evaluate_metrics.wrappers import ValueFunctWrapper, AutoMetricWrapper
from typing import List, Dict


def prepare_input(
        dialogue: List[Dict],
        max_context_len: int,
        response_field: str = 'response'
) -> List[Dict]:
    dialogue_turns = []
    for turn in dialogue:
        if turn['is_tracked']:
            new_turn = {
                'context': turn['context'][-max_context_len + 1:],
                'response': turn[response_field],
                'done': False
            }
            dialogue_turns.append(new_turn)
    dialogue_turns[-1]['done'] = True
    return dialogue_turns


def get_avg_reward(dialogue_batch: List[List[Dict]]):
    """
    Compute Average Retrun over dialogues. Get Return of a Dialogue.
    """
    rewards = []
    for dialogue in dialogue_batch:
        dial_reward = sum([x.get('reward', 0) for x in dialogue])
        rewards.append(dial_reward)
    return sum(rewards) / len(rewards)


def rate_convos(
        dialogue_batch: List[List[Dict]],
        auto_metric: AutoMetricWrapper,
        max_context_len: int
):
    rated_convos = []
    for dialogue in dialogue_batch:
        prepped_dials = prepare_input(dialogue, max_context_len)
        rewards = auto_metric.batch_rating(prepped_dials)

        for i in range(len(prepped_dials)):
            prepped_dials[i]['reward'] = float(rewards[i])

        rated_convos.append(prepped_dials)
    return rated_convos


def compute_advantages(
        dialogue_batch: List[List[Dict]],
        gamma: float
):
    for dialouge in dialogue_batch:
        for t in range(len(dialouge)):
            turn = dialouge[t]
            if not turn['done']:
                advantage = -turn['value'] + turn['reward'] + gamma*dialouge[t+1]['value']
            else:
                advantage = -turn['value'] + turn['reward'] #end of episode needs to be handled differently
            turn['advantage'] = advantage


def compute_values(
        dialogue_batch: List[List[Dict]],
        model: ValueFunctWrapper,
):
    for dialogue in dialogue_batch:
        values = model.batch_rating(dialogue)
        for i in range(len(dialogue)):
            dialogue[i]['value'] = float(values[i])
