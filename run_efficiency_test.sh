#!/bin/bash
sizes=( 1000 5000 10000 50000 100000 500000 1000000 )

for size in "${sizes[@]}"; do
    echo "Running efficiency test with sample size: $size"
    python -m src.droptc.efficiency_test --sample_size $size
done
