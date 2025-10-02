#!/bin/bash

python -m src.droptc.train_embedding --feature_col sentence --base_model all-mpnet-base-v2
python -m src.droptc.train_embedding --feature_col sentence --base_model all-MiniLM-L6-v2