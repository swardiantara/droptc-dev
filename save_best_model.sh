#!/bin/bash

# re-run the best models to save them with --save_model and --overwrite flag
# python -m src.droptc.train_classifier --scenario droptc --embedding bert-base-uncased --seed 52680723 --feature_col sentence --n_epochs 20 --save_model --overwrite
# python -m src.droptc.train_classifier --scenario droptc --embedding modern-bert --seed 87212562 --feature_col sentence --n_epochs 20 --save_model --overwrite
# python -m src.droptc.train_classifier --scenario droptc --embedding neo-bert --seed 14298463 --feature_col sentence --n_epochs 20 --save_model --overwrite
# python -m src.droptc.train_classifier --scenario droptc --embedding DroPTC-all-mpnet-base-v2-sentence --seed 87212562 --feature_col sentence --n_epochs 20 --save_model --overwrite
# python -m src.droptc.train_classifier --scenario droptc --embedding DroPTC-all-MiniLM-L6-v2-sentence --seed 99511865 --feature_col sentence --n_epochs 20 --save_model --overwrite

python -m src.droptc.train_classifier --scenario drolove --embedding bert-base-uncased --seed 90995999 --feature_col sentence --n_epochs 20 --save_model --overwrite
python -m src.droptc.train_classifier --scenario dronelog --embedding DroPTC-all-mpnet-base-v2-sentence --seed 14298463 --feature_col sentence --n_epochs 20 --save_model --overwrite
python -m src.droptc.train_classifier --scenario neurallog --embedding bert-base-uncased --seed 70681460 --feature_col sentence --n_epochs 20 --save_model --overwrite
python -m src.droptc.train_classifier --scenario transsentlog --embedding bert-base-uncased --seed 14298463 --feature_col sentence --n_epochs 20 --save_model --overwrite
echo "Best models have been re-trained and saved."