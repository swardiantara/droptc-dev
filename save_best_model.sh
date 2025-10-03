#!/bin/bash

# re-run the best models to save them with --save_model and --overwrite flag
python -m src.droptc.train_classifier --scenario droptc --embedding bert-base-uncased --seed 52680723 --feature_col sentence --n_epochs 20 --save_model --overwrite
python -m src.droptc.train_classifier --scenario droptc --embedding modern-bert --seed 87212562 --feature_col sentence --n_epochs 20 --save_model --overwrite
python -m src.droptc.train_classifier --scenario droptc --embedding neo-bert --seed 14298463 --feature_col sentence --n_epochs 20 --save_model --overwrite
python -m src.droptc.train_classifier --scenario droptc --embedding DroPTC-all-mpnet-base-v2-sentence --seed 87212562 --feature_col sentence --n_epochs 20 --save_model --overwrite
python -m src.droptc.train_classifier --scenario droptc --embedding DroPTC-all-MiniLM-L6-v2-sentence --seed 99511865 --feature_col sentence --n_epochs 20 --save_model --overwrite

# push the results
cd experiments/droptc
git add .
git commit -m "save best models after re-training for efficiency test"
git push
cd ../../..