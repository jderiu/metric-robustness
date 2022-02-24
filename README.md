# Code for the Paper: Probing the Robustness of Trained Metrics for Conversational Dialogue Systems

## Installation

- `pip install -r requirements.txt`

## Download Spacy and NLTK models:

- Spacy: `python -m spacy download en_core_web_sm`
- NLTK: `python -m nltk.downloader stopwords`

## Get the Models

- ATT: 
  - Download the model from https://github.com/golsun/AdversarialTuringTest
  - Store the hvm2.pth file to models/att_model/hvm2.pth
- Maude: 
  - https://github.com/facebookresearch/online_dialog_eval#getting-the-data
  - Store the distilbert_lm and na_all into models/maude_models/... 
    

## Train USR

To train the USR models, we need two training steps.

- Train the MLM model first: `python -m src.metrics.usr_metric.usr_train_mlm` 
  - The model is stored in models/usr_models/mlm_models/DOMAIN/
  - Note the checkpoint number and update in config/usr_train_config.json the "checkpoint_mlm_nr" accordingly.
    
- Train the Retrieval model: `python -m src.metrics.usr_metric.usr_train_retrieval` 
   - The model is stored in models/usr_models/ret_models/DOMAIN/
    
## Evaluate the Models on Correlation to Humans
You need to adapt the "checkpoint_mlm_nr" and "checkpoint_ret_nr" in config/eval_metric_config.json accodringly. 

`python -m src.evaulate_metrics.eval_auto_metric` 

## Run the RL 

You need to adapt the "checkpoint_mlm_nr" and "checkpoint_ret_nr" in config/rl_config.json accodringly. 

The file iterated over metrics and repeats the experiments 10 times. The seeds are fixed.

`bash run_rl_single_context.sh <domain> <blender_model>`

, where domain in {convai2, dailydialog, empathetic} and blender_model in {facebook/blenderbot-400M-distill, facebook/blenderbot_small-90M}.


## Evaluate Logs

`python -m src.reinforcement_learning.evaluate_logs` 
