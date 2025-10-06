#!/bin/bash
scenarios=( droptc drolove dronelog neurallog transsentlog )
models=( all-MiniLM-L6-v2 all-mpnet-base-v2 neo-bert modern-bert bert-base-uncased )
devices=( cpu cuda )
sizes=( 100 250 500 750 1000 2500 5000 10000 )
for scenario in "${scenarios[@]}"; do
    if [ "$scenario" = "droptc" ]; then
        for model in "${models[@]}"; do
            for device in "${devices[@]}"; do
                if [ "$device" = "cuda" ]; then
                    if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
                        echo "CUDA is not available. Skipping tests on GPU."
                        continue
                    fi
                fi
                for size in "${sizes[@]}"; do
                    echo "Running efficiency test with scenario: $scenario, model: $model, device: $device, sample size: $size"
                    python -m src.droptc.efficiency_test --scenario "$scenario" --model_name "$model" --device "$device" --sample_size $size --overwrite
                done
            done
        done
    fi
    if [ "$scenario" = "dronelog" ]; then
        model="all-mpnet-base-v2"
        for device in "${devices[@]}"; do
            if [ "$device" = "cuda" ]; then
                if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
                    echo "CUDA is not available. Skipping tests on GPU."
                    continue
                fi
            fi
            for size in "${sizes[@]}"; do
                echo "Running efficiency test with scenario: $scenario, model: $model, device: $device, sample size: $size"
                python -m src.droptc.efficiency_test --scenario "$scenario" --model_name "$model" --device "$device" --sample_size $size --overwrite
            done
        done
    else
        model="bert-base-uncased"
        for device in "${devices[@]}"; do
            if [ "$device" = "cuda" ]; then
                if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
                    echo "CUDA is not available. Skipping tests on GPU."
                    continue
                fi
            fi
            for size in "${sizes[@]}"; do
                echo "Running efficiency test with scenario: $scenario, model: $model, device: $device, sample size: $size"
                python -m src.droptc.efficiency_test --scenario "$scenario" --model_name "$model" --device "$device" --sample_size $size --overwrite
            done
        done
    fi
done
