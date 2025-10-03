#!/bin/bash
models=( neo-bert modern-bert bert-base-uncased )
devices=( cpu cuda )
sizes=( 1000 5000 10000 50000 100000 500000 1000000 )
for model in "${models[@]}"; do
    for device in "${devices[@]}"; do
        if [ "$device" = "cuda" ]; then
            if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
                echo "CUDA is not available. Skipping tests on GPU."
                continue
            fi
        fi
        for size in "${sizes[@]}"; do
            echo "Running efficiency test with model: $model, device: $device, sample size: $size"
            python -m src.droptc.efficiency_test --model_name "$model" --device "$device" --sample_size $size
        done
    done
done
