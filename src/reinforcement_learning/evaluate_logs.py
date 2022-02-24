import os, re
from collections import Counter
logging_path = 'logging/rl/rl-experiments-final-submitted'


print('domain\tmetric\tbl tpye\tBest Run\tBest Ep\tAvg Reward\tResp Freq\tLex Var\tBELU\tJacccard\tStrategy')
for domain in os.listdir(logging_path):
    logging_path_domain = os.path.join(logging_path, domain)
    for metric in os.listdir(logging_path_domain):
        for bl_type in ['small', 'large']:
            f_path = os.path.join(logging_path_domain, metric, 'len_30', bl_type)
            runs_data = []

            for run in os.listdir(f_path):
                run_nr = int(run.split('_')[-1])
                run_path = os.path.join(f_path, run)
                run_data = []
                for ofname in os.listdir(run_path):
                    ep = int(ofname.replace('.txt', '').split('_')[1])
                    with open(os.path.join(run_path, ofname), 'rt', encoding='utf-8') as ifile:
                        lines = ifile.readlines()
                        avg_reward = float(lines[0].replace('\n', '').split(' ')[-1])
                        response_freq = float(lines[1].replace('\n', '').split('\t')[0].split()[-1])
                        lex_variety = float(lines[1].replace('\n', '').split('\t')[1].split()[-1])
                        bleu = float(lines[1].replace('\n', '').split('\t')[2].split()[-1])
                        jaccard = float(lines[1].replace('\n', '').split('\t')[3].split()[-1])
                        run_data.append((ep, avg_reward, response_freq, lex_variety, bleu, jaccard))
                best_epoch = sorted(run_data, key=lambda x: x[1], reverse=True)[0]
                runs_data.append((run_nr, best_epoch))
            if len(runs_data) == 0:
                print(f'{domain}\t{metric}\t{bl_type}\t NO DATA')
                continue
            best_run = sorted(runs_data, key=lambda x: x[1][1], reverse=True)[0]
            best_strategy = 'Not Conclusive'
            fixed_response = ''
            if best_run[1][2] > 0.75:
                best_strategy = 'Fixed Response'
                run_path = os.path.join(f_path, f'run_{best_run[0]}', f'outs_{best_run[1][0]}.txt')
                with open(run_path, 'rt', encoding='utf-8') as ifile:
                    lines = ifile.readlines()
                    responses = [re.split('\(.*\)', line.replace('\n', ''))[-1] for line in lines[2:]]
                    c = Counter(responses)
                    fixed_response = c.most_common(1)[0][0]

            elif best_run[1][4] > 0.2:
                best_strategy = 'Parrot'
            elif best_run[1][2] < 0.1 and best_run[1][3] < 0.1 and best_run[1][5] > 0.05:
                best_strategy = 'Pattern'
            #print(domain, metric, bl_type, best_run)
            print(f'{domain}\t{metric}\t{bl_type}\t{best_run[0]}\t{best_run[1][0]}\t{best_run[1][1]}\t{best_run[1][2]}\t{best_run[1][3]}\t{best_run[1][4]}\t{best_run[1][5]}\t{best_strategy}\t{fixed_response}')
