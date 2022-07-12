import json, itertools
from collections import defaultdict
from src.evaluate_metrics.wrappers import *
from src.utils.data_preprocessing import AutoMetricData
from tqdm import tqdm
import numpy as np
import scipy.stats


def ranking_correlation(ratings):
    sample_to_data = defaultdict(lambda: [])
    for did, tid, system_name, human_rating, pred_rating in ratings:
        sample_to_data[(did, tid)].append((system_name, human_rating, pred_rating))

    ranking_for_sample_human = defaultdict(lambda: [0, 0])
    ranking_for_sample_pred = defaultdict(lambda: [0, 0])
    for key, data in sample_to_data.items():
        for d0, d1 in itertools.combinations(data, r=2):
            system_name0, human_rating0, pred_rating0 = d0
            system_name1, human_rating1, pred_rating1 = d1
            ranking_for_sample_human[system_name0][1] += 1
            ranking_for_sample_human[system_name1][1] += 1
            ranking_for_sample_pred[system_name0][1] += 1
            ranking_for_sample_pred[system_name1][1] += 1

            if human_rating0 > human_rating1:
                ranking_for_sample_human[system_name0][0] += 1
            elif human_rating0 < human_rating1:
                ranking_for_sample_human[system_name1][0] += 1

            if pred_rating0 > pred_rating1:
                ranking_for_sample_pred[system_name0][0] += 1
            elif pred_rating0 < pred_rating1:
                ranking_for_sample_pred[system_name1][0] += 1

    win_rates = {system_type: x[0] / x[1] for system_type, x in ranking_for_sample_human.items()}
    ranking_human = [x[0] for x in sorted(win_rates.items(), key=lambda x: x[1], reverse=True)]

    win_rates = {system_type: x[0] / x[1] for system_type, x in ranking_for_sample_pred.items()}
    ranking_pred = [x[0] for x in sorted(win_rates.items(), key=lambda x: x[1], reverse=True)]

    tau, p_val = scipy.stats.kendalltau(ranking_human, ranking_pred)
    return tau


def corr_to_humans(auto_metric, data, eval_batch_size, domain, metric_name):
    ratings = []
    iterations = int(len(data.db_samples) / eval_batch_size + 1)
    for batched_sample in tqdm(batch_samples(data.db_samples, eval_batch_size), total=iterations, desc=f'Compute {domain} {metric_name}'):
        avg_per_sample_mlm_loss = auto_metric.batch_rating(batched_sample)
        for i in range(len(batched_sample)):
            dialogue_id = batched_sample[i]['dialogue_id']
            turn_id = batched_sample[i]['turn_id']
            system_name = batched_sample[i]['system_name']
            avg_rating = batched_sample[i]['avg_rating']
            ratings.append((
                dialogue_id,
                turn_id,
                system_name,
                avg_rating,
                float(avg_per_sample_mlm_loss[i])))

    hj = np.array([x[3] for x in ratings])
    losses = np.array([x[4] for x in ratings])
    # print(len(ratings), hj.shape, losses.shape)
    pearson = scipy.stats.pearsonr(hj, losses)
    spearman = scipy.stats.spearmanr(hj, losses)

    return pearson, spearman, ratings


def corr_to_humans_eval(auto_metric, data, domain, metric_name):
    pearson, spearman, ratings = corr_to_humans(auto_metric, data, eval_batch_size, domain, metric_name)

    scores_for_system = defaultdict(lambda: [])
    for did, tid, system_name, human_rating, pred_rating in ratings:
        scores_for_system[system_name].append((human_rating, pred_rating))

    human_for_system, predicted_for_system = {}, {}
    for system_name, system_ratings in sorted(scores_for_system.items()):
        avg_human_rating = sum([x[0] for x in system_ratings]) / len(system_ratings)
        avg_pred_rating = sum([x[1] for x in system_ratings]) / len(system_ratings)
        human_for_system[system_name] = avg_human_rating
        predicted_for_system[system_name] = avg_pred_rating

    human_for_system_sorted = [x[0] for x in sorted(human_for_system.items(), key=lambda x: x[1], reverse=True)]
    predicted_for_system_sorted = [x[0] for x in sorted(predicted_for_system.items(), key=lambda x: x[1], reverse=True)]

    tau, p_val = scipy.stats.kendalltau(human_for_system_sorted, predicted_for_system_sorted)

    tau_comp = ranking_correlation(ratings)

    logging_path = os.path.join(f"logging/eval_auto_metric/{domain}")
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    with open(os.path.join(logging_path, f'output_{metric_name}.txt'), 'wt', encoding='utf-8') as lfile:
        lfile.write(f'Pearson: {pearson[0]}\tSpearman: {spearman}\n')
        for x in ratings:
            oline = f'{x[0]}\t{x[1]}\t{x[2]}\t{x[3]}\t{x[4]}\n'
            lfile.write(oline)

    return spearman[0], pearson[0], tau, tau_comp, scores_for_system


def batch_samples(samples, eval_batch_size):
    for idx in range(0, len(samples), eval_batch_size):
        yield samples[idx:idx + eval_batch_size]


def print_str(data_for_metric_domain, val, metrics):
    out_string = '\tDD\tED\tPC\n'
    for metric in metrics:
        line = f'{metric}'
        for domain in domains:
            spearman, pearson, tau, tau_comp, scores_for_system = data_for_metric_domain[domain, metric]
            if val == 'spearman':
                line += f'\t{spearman:.4f}'
            elif val == 'pearson':
                line += f'\t{pearson:.4f}'
            elif val == 'tau':
                line += f'\t{tau:.4f}'
            elif val == 'tau_comp':
                line += f'\t{tau_comp:.4f}'
        out_string += f'{line}\n'
    return out_string


def print_scores(data_for_metric_domain, all_metrics):
    out_string = ''
    metrics_header = '\t'.join(all_metrics)
    for domain in domains:
        out_string += f'{domain}\n'
        out_string += f'\tAMT\t{metrics_header}\n'
        system_names = sorted(data_for_metric_domain[domain, all_metrics[0]][-1].keys())
        for system_name in system_names:
            system_ratings = data_for_metric_domain[domain, all_metrics[0]][-1][system_name]
            avg_human_rating = sum([x[0] for x in system_ratings]) / len(system_ratings)

            out_string += f'{system_name}\t{avg_human_rating:.4f}'
            for metric in all_metrics:
                _, _, _, _, scores_for_system = data_for_metric_domain[domain, metric]
                avg_score = sum([x[1] for x in scores_for_system[system_name]])/len( scores_for_system[system_name])
                out_string += f'\t{avg_score:.4f}'
            out_string += '\n'
        out_string += '\n'

    return out_string


def perform_single_metric_test(metrics: List[str], domains: List[str]):
    data_for_metric_domain = {}
    for domain in domains:
        data = AutoMetricData(domain)
        checkpoint_mlm_nr = config['checkpoint_mlm_nr'][domain]
        checkpoint_ret_nr = config['checkpoint_ret_nr'][domain]

        for metric in metrics:
            auto_metric = get_metric(metric, domain, checkpoint_ret_nr=checkpoint_ret_nr, checkpoint_mlm_nr=checkpoint_mlm_nr, device=device)
            spearman, pearson, tau, tau_comp, scores_for_system = corr_to_humans_eval(auto_metric, data, domain, metric)
            data_for_metric_domain[domain, metric] = (spearman, pearson, tau, tau_comp, scores_for_system)

    out_spearman = print_str(data_for_metric_domain, 'spearman', metrics)
    out_tau = print_str(data_for_metric_domain, 'tau', metrics)
    out_tau_comp = print_str(data_for_metric_domain, 'tau_comp', metrics)
    out_pearson = print_str(data_for_metric_domain, 'pearson', metrics)

    print(out_spearman + '\n')
    print(out_tau + '\n')
    print(out_tau_comp + '\n')
    print(out_pearson + '\n')

    out_rankings = print_scores(data_for_metric_domain, metrics)
    print(out_rankings)


def perform_degenerate_strategy_test(metrics: List[str], domains: List[str]):
    data_for_metric_domain = {}
    for domain in domains:
        data = AutoMetricData(domain)
        for metric in metrics:
            data.load_db_samples() #need to reload, otherwise we add too much
            data.add_parrot_str()
            data.add_pattern()
            data.add_policy_parrot()
            data.add_fixed_response_bot(metric)

            auto_metric = get_metric(metric, domain, checkpoint_ret_nr=checkpoint_ret_nr, checkpoint_mlm_nr=checkpoint_mlm_nr, device=device)
            spearman, pearson, tau, tau_comp, scores_for_system = corr_to_humans_eval(auto_metric, data, domain, metric)
            data_for_metric_domain[domain, metric] = (spearman, pearson, tau, tau_comp, scores_for_system)

    out_spearman = print_str(data_for_metric_domain, 'spearman', metrics)
    out_tau = print_str(data_for_metric_domain, 'tau', metrics)
    out_tau_comp = print_str(data_for_metric_domain, 'tau_comp', metrics)
    out_pearson = print_str(data_for_metric_domain, 'pearson', metrics)

    print(out_spearman + '\n')
    print(out_tau + '\n')
    print(out_tau_comp + '\n')
    print(out_pearson + '\n')

    out_rankings = print_scores(data_for_metric_domain, metrics)
    print(out_rankings)


if __name__ == '__main__':
    with open('config/eval_metric_config.json', 'rt', encoding='utf-8') as ifile:
        config = json.load(ifile)

    eval_batch_size = config['eval_batch_size']
    max_length = config['max_len']
    device = config['device']

    _metrics = ['usr_ret', 'usr_mlm', 'usr_full_reg', 'att', 'maude',  'blender']
    domains = ['dailydialog', 'empathetic', 'convai2']

    perform_single_metric_test(_metrics, domains)
    perform_degenerate_strategy_test(_metrics, domains)



