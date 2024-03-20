import os, random


class DataLoader:
    def __init__(self, domain, seed):
        self.data_path = f'data/usr_train_data/{domain}'
        self.seed = seed
        valid_sample_n, test_sample_n = 0, 1000
        self.valid_seeds = self.prepare_seeds('valid_none_original.txt', valid_sample_n)
        self.test_seeds = self.prepare_seeds('test_none_original.txt', test_sample_n)

    def prepare_seeds(self, fname, sample_n=0):
        all_seeds = []
        with open(os.path.join(self.data_path, fname), 'rt', encoding='utf-8') as ifile:
            current_convo = []
            for line in ifile.readlines():
                tid = int(line.split()[0])
                if tid == 1: # start of new convo
                    for t0, t1 in zip(current_convo, current_convo[1:]):
                        all_seeds.append([t0, t1])
                    current_convo = []
                line = line.replace('\n', '')


                turn0 = ' '.join(line.split('\t')[0].split()[1:])
                turn1 = line.split('\t')[1]
                current_convo.extend([turn0, turn1])
        if sample_n > 0:
            random.seed(23432) #make sure it's always the same set
            all_seeds = random.sample(all_seeds, k=sample_n)
            random.seed(self.seed)
        return all_seeds