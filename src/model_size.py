import os
import json
import torch
from collections import defaultdict
from src.droptc.efficiency_test import get_embedding_model
from src.droptc.train_classifier import MODEL_REGISTRY

def main():
    scenario2model = {
        'droptc-minilm': 'sentence-transformers/all-MiniLM-L6-v2',
        'droptc-mpnet': 'sentence-transformers/all-mpnet-base-v2',
        'droptc-bert': 'bert-base-uncased',
        'droptc-neo': 'chandar-lab/NeoBERT',
        'droptc-modern': 'answerdotai/ModernBERT-base',
        'neurallog': 'bert-base-uncased',
        'drolove': 'bert-base-uncased',
        'dronelog': 'sentence-transformers/all-mpnet-base-v2',
        'transsentlog': 'bert-base-uncased'
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_size = defaultdict(lambda: defaultdict(int))
    for scenario, model in scenario2model.items():
        embedding_model, tokenizer = get_embedding_model(model, device)
        
        model_class = MODEL_REGISTRY['droptc']
        model = model_class(embedding_model, tokenizer, freeze_embedding=False).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size[scenario]['total_params'] = total_params
        model_size[scenario]['trainable_params'] = trainable_params
        print(f"Model: {model.__class__.__name__}, Total Params: {total_params}, Trainable Params: {trainable_params}")

    with open(os.path.join('experiments', 'analysis', 'model_size.json'), 'w') as f:
        json.dump(model_size, f, indent=4)

        
if __name__ == "__main__":
    main()