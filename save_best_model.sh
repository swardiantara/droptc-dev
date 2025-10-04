#!/bin/bash

# re-run the best models to save them with --save_model and --overwrite flag
python -m src.droptc.train_classifier --scenario droptc --embedding bert-base-uncased --seed 52680723 --feature_col sentence --n_epochs 20 --save_model --overwrite
cp experiments/droptc/sentence/bert-base-uncased/unfreeze/52680723/sentence_pytorch_model.pt src/cli/model/pytorch_model_bert.pt
python -m src.droptc.train_classifier --scenario droptc --embedding modern-bert --seed 87212562 --feature_col sentence --n_epochs 20 --save_model --overwrite
cp experiments/droptc/sentence/modern-bert/unfreeze/87212562/sentence_pytorch_model.pt src/cli/model/pytorch_model_modern.pt
python -m src.droptc.train_classifier --scenario droptc --embedding neo-bert --seed 14298463 --feature_col sentence --n_epochs 20 --save_model --overwrite
cp experiments/droptc/sentence/neo-bert/unfreeze/14298463/sentence_pytorch_model.pt src/cli/model/pytorch_model_neo.pt
python -m src.droptc.train_classifier --scenario droptc --embedding DroPTC-all-mpnet-base-v2-sentence --seed 87212562 --feature_col sentence --n_epochs 20 --save_model --overwrite
cp experiments/droptc/sentence/DroPTC-all-mpnet-base-v2-sentence/unfreeze/87212562/sentence_pytorch_model.pt src/cli/model/pytorch_model_mpnet.pt
python -m src.droptc.train_classifier --scenario droptc --embedding DroPTC-all-MiniLM-L6-v2-sentence --seed 99511865 --feature_col sentence --n_epochs 20 --save_model --overwrite
cp experiments/droptc/sentence/DroPTC-all-MiniLM-L6-v2-sentence/unfreeze/99511865/sentence_pytorch_model.pt src/cli/model/pytorch_model.pt

python -m src.droptc.train_classifier --scenario drolove --embedding bert-base-uncased --seed 90995999 --feature_col sentence --n_epochs 20 --save_model --overwrite
cp experiments/drolove/sentence/bert-base-uncased/unfreeze/90995999/sentence_pytorch_model.pt src/cli/model/pytorch_model_drolove.pt
python -m src.droptc.train_classifier --scenario dronelog --embedding DroPTC-all-mpnet-base-v2-sentence --seed 14298463 --feature_col sentence --n_epochs 20 --save_model --overwrite
cp experiments/dronelog/sentence/DroPTC-all-mpnet-base-v2-sentence/unfreeze/14298463/sentence_pytorch_model.pt src/cli/model/pytorch_model_dronelog.pt
python -m src.droptc.train_classifier --scenario neurallog --embedding bert-base-uncased --seed 70681460 --feature_col sentence --n_epochs 20 --save_model --overwrite
cp experiments/neurallog/sentence/bert-base-uncased/unfreeze/70681460/sentence_pytorch_model.pt src/cli/model/pytorch_model_neurallog.pt
python -m src.droptc.train_classifier --scenario transsentlog --embedding bert-base-uncased --seed 14298463 --feature_col sentence --n_epochs 20 --save_model --overwrite
cp experiments/transsentlog/sentence/bert-base-uncased/unfreeze/14298463/sentence_pytorch_model.pt src/cli/model/pytorch_model_transsentlog.pt

echo "Best models have been re-trained and saved."