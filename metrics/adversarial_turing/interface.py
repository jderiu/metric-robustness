from auto_judge.adversarial_turing.main import Option
from auto_judge.adversarial_turing.scorer import load_scorer


class DummyArgs():
    def __init__(self, cpu):
        self.cpu = cpu

    task = 'play'
    cpu = False
    batchG = 32
    batchD = 32
    vali_size = 128
    vali_print = 3
    lrG = 3e-5
    lrD = 1e-4
    step_vali = 50
    max_l_cxt = 60
    max_l_rsp = 30
    max_n_hyp = 10
    wt_rl = 1
    debug = False
    switchD = 3000
    switchG = 1000
    accG = 0.5
    accD = 0.7
    csize = 1000
    T = 0.0
    verbose = False
    dataG = ''
    dataD = ''
    prev_dataD = 'nää'
    last = False
    path_scorer = 'auto_judge/adversarial_turing/restore/hvm2.pth'
    path_gen = ''


class AdversarialTuring():
    def __init__(self, cpu):

        opt = Option(DummyArgs(cpu))
        self.scorer = load_scorer(opt)

    def get_score(self, context:str, hypothesis:str):
        return self.scorer.predict(context, [hypothesis])

    def eval(self):
        self.scorer.eval()