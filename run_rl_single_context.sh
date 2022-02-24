##!/usr/bin/env bash
export PYHTONPATH=.

apps=(0 1 2 3 4 5 6 7 8 9)
seeds=(1232132 555432 878923 3270 43420 3542343 35478 65343 72344)
metrics=('usr_ret' 'usr_mlm' 'usr_full_reg' 'att' 'maude'  'blender')

if [[ $2 == 'facebook/blenderbot-400M-distill' ]]; then
  batch_size=2
  dial_batch_size=2
elif [[ $2 == 'facebook/blenderbot_small-90M' ]]; then
  batch_size=8
  dial_batch_size=8
fi

for app in "${apps[@]}"; do
		for metric in "${metrics[@]}"; do
			python -m src.reinforcement_learning.run_code -d $1 -m ${metric} -b $2 -l 50 -a ${app} --batch_size ${batch_size} --dial_batch_size ${dial_batch_size} --seed ${seeds[${app}]}
		done
done
