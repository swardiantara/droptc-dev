import torch

import numpy as np

from typing import Tuple
from captum.attr import visualization


label2idx = {
    'Normal': 0,
    'SurroundingEnvironment': 1,
    'HardwareFault': 2,
    'ParamViolation': 3,
    'RegulationViolation': 4,
    'CommunicationIssue': 5,
    'SoftwareFault': 6
}

idx2label = {
    0: 'Normal',
    1: 'SurroundingEnvironment',
    2: 'HardwareFault',
    3: 'ParamViolation',
    4: 'RegulationViolation',
    5: "CommunicationIssue",
    6: 'SoftwareFault'
}

raw2pro = {
    'normal': 'Normal',
    'SurEnv': 'SurroundingEnvironment',
    'HwFlt': 'HardwareFault',
    'ConfIss': 'ParamViolation',
    'VioReg': 'RegulationViolation',
    'CommIss': "CommunicationIssue",
    'Swflt': 'SoftwareFault',
}


class2color = {
    'normal': '#4CAF50',
    'low': '#FFC107',
    'medium': '#FF5722',
    'high': '#FF5722', 
}


def reconstruct_roberta_tokens(tokens, attributions):
    words = []
    attribution_score = []
    current_word = ""
    current_attr = 0
    for idx, (token, attribution) in enumerate(zip(tokens, attributions)):
        if idx < 2:
            if current_word:
                words.append(current_word)
                attribution_score.append(current_attr)
            current_word = token
            current_attr = attribution
        elif idx == len(tokens) - 1:
            if current_word:
                words.append(current_word)
                attribution_score.append(current_attr)
            current_word = token
            current_attr = attribution
        elif 'Ġ' in token and len(token) > 1: # beginning of new word
            if current_word:
                words.append(current_word)
                attribution_score.append(current_attr)
            current_word = token[1:]
            current_attr = attribution
        elif token == 'Ġ':
            continue # ignore this token
        elif len(token) == 1: # punctuation
            if current_word:
                words.append(current_word)
                attribution_score.append(current_attr)
            current_word = token
            current_attr = attribution
        else:
            current_word += token
            current_attr += attribution
            # if current_word:
            #     words.append(current_word)
            #     attribution_score.append(current_attr)
            # current_word = token
            # current_attr = attribution
    # Append the last word
    if current_word:
        words.append(current_word)
        attribution_score.append(current_attr)

    return words, attribution_score


def reconstruct_tokens(tokens, attributions):
    words = []
    attribution_score = []
    current_word = ""
    current_attr = 0
    for token, attribution in zip(tokens, attributions):
        if token.startswith("##"):
            current_word += token[2:]  # Remove "##" and append to the current word
            current_attr += attribution
        else:
            if current_word:
                words.append(current_word)
                attribution_score.append(current_attr)
            current_word = token
            current_attr = attribution
    # Append the last word
    if current_word:
        words.append(current_word)
        attribution_score.append(current_attr)

    return words, attribution_score


def infer_pred(model, input_ids, attention_mask, idx2label)-> Tuple[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        pred_prob = torch.softmax(logits, dim=1)  # 'dim' is preferred over 'axis' in PyTorch

        # Get predicted class index (int)
        pred_idx = torch.argmax(pred_prob, dim=1).item()

        # Convert class index to label
        pred_label = idx2label.get(pred_idx, 'Normal')

        # Get predicted probability value (float)
        pred_prob_val = pred_prob[0, pred_idx].item()

    return pred_label, pred_prob_val  # (str, float)


def scale_attribution(distribution):
    """
    Scales the input distribution to the range [-1, 1].

    Parameters:
    distribution (numpy.ndarray): The input distribution of values to be scaled.

    Returns:
    numpy.ndarray: The scaled distribution with values in the range [-1, 1].
    """
    distribution = np.asarray(distribution)
    min_val = np.min(distribution)
    max_val = np.max(distribution)
    scaled_distribution = 2 * (distribution - min_val) / (max_val - min_val) - 1
    return scaled_distribution


def get_embedding_layer(model):
    """
    Identifies and returns the embedding layer for a given Hugging Face model.
    """
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        model_type = model.config.model_type
        print(f"Detected model_type from config: {model_type}")

        if model_type == "neobert":
            # Based on your print(model) output for NeoBERT
            return model.encoder
        elif model_type == "bert":
            # Standard BERT model
            return model.embeddings
        elif model_type == "minilm":
            # For MiniLM, it often follows the BERT structure
            return model.embeddings
        elif model_type == "modernbert":
            # For MiniLM, it often follows the BERT structure
            return model.embeddings
        # Add more conditions for other model types if needed
        # You would need to inspect the structure of each new model type
        # using print(model) to find the correct path.
        else:
            print(f"Warning: Model type '{model_type}' not explicitly handled for embedding layer. Attempting common paths.")
            # Fallback for other common architectures
            if hasattr(model, 'embeddings'):
                return model.embeddings
            else:
                raise AttributeError(f"Could not find a common embedding layer for model type: {model_type}")
    else:
        print("Warning: Model does not have a standard config.model_type. Attempting common paths based on inspection.")
        # Fallback if config.model_type isn't available
        if hasattr(model, 'encoder') and isinstance(model.encoder, torch.nn.Embedding):
            return model.encoder
        elif hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
            return model.bert.embeddings
        elif hasattr(model, 'embeddings'):
            return model.embeddings
        else:
            raise AttributeError("Could not find a common embedding layer. Please inspect the model structure.")


def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, delta):
    attributions = np.array(attributions)
    # storing couple samples in an array for visualization purposes
    return visualization.VisualizationDataRecord(
                            word_attributions=attributions,
                            pred_prob=pred,
                            pred_class=pred_ind,
                            true_class=label, # true label
                            attr_class=pred_ind, # attribution label
                            attr_score=attributions.sum(),
                            raw_input_ids=text,
                            convergence_score=delta)


def to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64, torch.Tensor)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj
