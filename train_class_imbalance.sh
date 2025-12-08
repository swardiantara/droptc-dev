#!/bin/bash
word_embeds=( all-mpnet-base-v2 all-MiniLM-L6 )
freezes=( false )
seeds=( 14298463 24677315 37622020 43782163 52680723 67351593 70681460 87212562 90995999 99511865 )
losses=( ce )
class_weights=( balanced inverse )
for embedding in "${word_embeds[@]}"; do
    for loss in "${losses[@]}"; do
        for class_weight in "${class_weights[@]}"; do
            for seed in "${seeds[@]}"; do
                for freeze in "${freezes[@]}"; do
                    if [ "$freeze" = true ]; then
                        python -m src.droptc.train_classifier --embedding "$embedding" --seed "$seed" --feature_col sentence --n_epochs 20 --loss_fn "$loss" --class_weight "$class_weight" --freeze_embedding 
                    else
                        python -m src.droptc.train_classifier --embedding "$embedding" --seed "$seed" --feature_col sentence --n_epochs 20 --loss_fn "$loss" --class_weight "$class_weight"
                    fi
                done
            done
        done
    done
done
