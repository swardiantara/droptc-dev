#!/bin/bash
# word_embeds=( bert-base-uncased neo-bert modern-bert all-MiniLM-L6-v2 all-mpnet-base-v2 DroPTC-all-mpnet-base-v2-sentence DroPTC-all-MiniLM-L6-v2-sentence )
# freezes=( true false )
# seeds=( 14298463 24677315 37622020 43782163 52680723 67351593 70681460 87212562 90995999 99511865 )

# for embedding in "${word_embeds[@]}"; do
#     for seed in "${seeds[@]}"; do
#         for freeze in "${freezes[@]}"; do
#             if [ "$freeze" = true ]; then
#                 python -m src.droptc.train_classifier --embedding "$embedding" --seed "$seed" --feature_col sentence --n_epochs 20 --freeze_embedding 
#             else
#                 python -m src.droptc.train_classifier --embedding "$embedding" --seed "$seed" --feature_col sentence --n_epochs 20
#             fi
#         done
#     done
# done

# DroneLog
word_embeds=( DroPTC-all-mpnet-base-v2-sentence )
freezes=( false )
seeds=( 14298463 24677315 37622020 43782163 52680723 67351593 70681460 87212562 90995999 99511865 )
for embedding in "${word_embeds[@]}"; do
    for seed in "${seeds[@]}"; do
        for freeze in "${freezes[@]}"; do
            if [ "$freeze" = true ]; then
                python -m src.droptc.train_classifier --embedding "$embedding" --scenario dronelog --seed "$seed" --feature_col sentence --n_epochs 20 --freeze_embedding --overwrite
            else
                python -m src.droptc.train_classifier --embedding "$embedding" --scenario dronelog --seed "$seed" --feature_col sentence --n_epochs 20 --overwrite
            fi
        done
    done
done

# NeuralLog
# word_embeds=( bert-base-uncased )
# freezes=( false )
# seeds=( 14298463 24677315 37622020 43782163 52680723 67351593 70681460 87212562 90995999 99511865 )
# for embedding in "${word_embeds[@]}"; do
#     for seed in "${seeds[@]}"; do
#         for freeze in "${freezes[@]}"; do
#             if [ "$freeze" = true ]; then
#                 python -m src.droptc.train_classifier --embedding "$embedding" --scenario neurallog --seed "$seed" --feature_col sentence --n_epochs 20 --freeze_embedding 
#             else
#                 python -m src.droptc.train_classifier --embedding "$embedding" --scenario neurallog --seed "$seed" --feature_col sentence --n_epochs 20
#             fi
#         done
#     done
# done

# # TransSentLog
# word_embeds=( bert-base-uncased )
# freezes=( false )
# seeds=( 14298463 24677315 37622020 43782163 52680723 67351593 70681460 87212562 90995999 99511865 )
# for embedding in "${word_embeds[@]}"; do
#     for seed in "${seeds[@]}"; do
#         for freeze in "${freezes[@]}"; do
#             if [ "$freeze" = true ]; then
#                 python -m src.droptc.train_classifier --embedding "$embedding" --scenario transsentlog --seed "$seed" --feature_col sentence --n_epochs 20 --freeze_embedding 
#             else
#                 python -m src.droptc.train_classifier --embedding "$embedding" --scenario transsentlog --seed "$seed" --feature_col sentence --n_epochs 20
#             fi
#         done
#     done
# done

# # DroLoVe
# word_embeds=( bert-base-uncased )
# freezes=( false )
# seeds=( 14298463 24677315 37622020 43782163 52680723 67351593 70681460 87212562 90995999 99511865 )
# for embedding in "${word_embeds[@]}"; do
#     for seed in "${seeds[@]}"; do
#         for freeze in "${freezes[@]}"; do
#             if [ "$freeze" = true ]; then
#                 python -m src.droptc.train_classifier --embedding "$embedding" --scenario drolove --seed "$seed" --feature_col sentence --n_epochs 20 --freeze_embedding 
#             else
#                 python -m src.droptc.train_classifier --embedding "$embedding" --scenario drolove --seed "$seed" --feature_col sentence --n_epochs 20
#             fi
#         done
#     done
# done