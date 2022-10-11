import json
import random
import spacy

fix_response_lookup = {
    'usr_ret': {
        'convai2': 'I love to be a musician.   I love music.  What kind of music do you listen to as a music lover',
        'dailydialog': '',
        'empathetic': '',
    },
    'usr_mlm': {
        'convai2': 'i am a stay at home mom and i am trying to figure out what i want to do with my life',
        'dailydialog': 'i am not sure if i am going to be able to go out of my way to get to know each other or not.',
        'empathetic': "I don’t know what I'm going to do if it doesn’t work out. I’m not sure what to do.",
    },
    'usr_full_reg': {
        'convai2': 'I love to do things with my time. What kind of things do you enjoy to do with your time? I love all kinds',
        'dailydialog': '',
        'empathetic': "",
    },
    'att': {
        'convai2': "yea!!! 1!! 2!! 3!! * * * fucking fucking fucking * * [ [ [ fucking * fucking *",
        'dailydialog': "! freaking out! one of these days! * * one * * freaking * * out! * even * * damn * * even damn",
        'empathetic': "I know right? I felt SO SO ASHAmed of myself. I felt so embar assed.",
    },
    'maude': {
        'convai2': "What kind of work do you have?  What do you like to do in your free time?",
        'dailydialog': "I hope it works out for you.  What kind of car did you get?",
        'empathetic': "I wish I could go back in time and be a kid again. I miss those days. ",
    },
    'blender': {
        'convai2': "that sounds like a lot of fun. what do you like to do in your spare time?",
        'dailydialog': "that would be great! what do you do for a living, if you don't mind me asking?",
        'empathetic': "I'm so sorry to hear that. What happened, if you don't mind me asking?",
    }
}

class AutoMetricData():
    def __init__(self, domain: str):
        self.data_path = f'auto_judge/USR/{domain}'

        self.context_len = 4
        self.domain = domain

        self.load_db_samples()

    def load_db_samples(self):
        with open(f'data/annotated_data/annotated_data_{self.domain}.json', 'rt', encoding='utf-8') as ifile:
            self.db_samples = json.load(ifile)

    def dumb_samples(self):
        with open(f'data/annotated_data/annotated_rated_data_{self.domain}.json', 'wt', encoding='utf-8') as ifile:
             json.dump(self.db_samples, ifile)

    def prepare_dicts(self):
        turns = set()
        for entry in self.db_samples:
            dialogue_id = entry['dialogue_id']
            turn_id = entry['turn_id']
            turns.add((dialogue_id, turn_id))

        prepped_dicts = []
        for dialogue_id, turn_id in turns:
            d = [x for x in self.db_samples if x['dialogue_id'] == dialogue_id and x['turn_id'] == turn_id and x['system_name'] == 'human'][0]
            prepped_dicts.append(d.copy())
        return prepped_dicts

    def add_policy_parrot(self):
        with open(f'data/annotated_data/generated_parrot_policy_{self.domain}.json', 'rt', encoding='utf-8') as ifile:
            parrot_samples = json.load(ifile)
            for sample in parrot_samples:
                sample['response'] = sample['response'][0]

            self.db_samples.extend(parrot_samples)

    def add_parrot_str(self):
        prepped_dicts = self.prepare_dicts()
        for prepped_dict in prepped_dicts:
            response = ' '.join(prepped_dict['context'])

            prepped_dict['system_name'] = 'parrot_bot'
            prepped_dict['response'] = response
            prepped_dict['avg_rating'] = 0.0

            self.db_samples.append(prepped_dict)

    def add_pattern(self):
        if self.domain == 'empathetic':
            self.add_pattern_empathetic()
        elif self.domain == 'dailydialog':
            self.add_pattern_daily()

    def add_pattern_daily(self):
        nlp = spacy.load('en_core_web_sm')
        fix_response = "i'm not sure if i'd like to {}. i'll let you know if i do."
        prepped_dicts = self.prepare_dicts()
        for prepped_dict in prepped_dicts:

            context = prepped_dict['context']
            tokens = nlp(" ".join(context))

            phrase = tokens[:20]

            prepped_dict['system_name'] = 'pattern_bot'
            prepped_dict['response'] = fix_response.format(phrase)
            prepped_dict['avg_rating'] = 0.0

            self.db_samples.append(prepped_dict)

    def add_pattern_empathetic(self):
        nlp = spacy.load('en_core_web_sm')
        fix_response = "i don't think it's {}. i'm sorry to hear that. what do you mean by that?"
        prepped_dicts = self.prepare_dicts()

        for prepped_dict in prepped_dicts:

            context = prepped_dict['context']
            tokens = [token.text for token in nlp(" ".join(context)) if token.pos_ == 'NOUN']
            if len(tokens) == 0:
                noun = 'that'
            else:
                noun = random.choice(tokens)

            prepped_dict['system_name'] = 'pattern_bot'
            prepped_dict['response'] = fix_response.format(noun)
            prepped_dict['avg_rating'] = 0.0

            self.db_samples.append(prepped_dict)

    def add_fixed_response_bot(self, metric):
        fix_response = fix_response_lookup[metric][self.domain]
        prepped_dicts = self.prepare_dicts()

        for prepped_dict in prepped_dicts:
            prepped_dict['system_name'] = 'fix_response_bot'
            prepped_dict['response'] = fix_response
            prepped_dict['avg_rating'] = 0.0

            self.db_samples.append(prepped_dict)
