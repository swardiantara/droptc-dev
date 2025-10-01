import os
import json
import torch
import pdfkit
import pandas as pd

from os import name
from datetime import datetime

from captum.attr import LayerIntegratedGradients, visualization
from src.droptc.interpretability import get_embedding_layer, infer_pred, reconstruct_roberta_tokens, reconstruct_tokens, add_attributions_to_visualizer, to_serializable
from src.droptc.train_classifier import pro2idx, idx2pro, raw2pro  

def load_config(config_path: str) -> dict:
    """
    Load a JSON configuration file.

    Args:
        config_path: The path to the JSON configuration file.

    Returns:
        A dictionary containing the configuration settings.
    """

    with open(config_path, 'r') as file:
        config = json.load(file)
    
    return config



def create_workdir(config_file: dict) -> str:
    """
    Create a working directory based on the current timestamp inside the specified output directory.

    Args:
        config_file: A dictionary containing configuration settings, including 'output_dir'.

    Returns:
        The path to the created working directory.
    """
    # Ensure the base output directory exists
    os.makedirs(config_file['output_dir'], exist_ok=True)

    # Create a timestamped subdirectory
    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config_file['output_dir'], now)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def extract_message(evidence_file: str) -> list[str]:
    """
    Extracts the 'timestamp' and the 'message' column from a CSV evidence file.

    Args:
        evidence_file: Path to the CSV file containing the evidence data.
    Returns:
        Pandas dataframe containing the timestamp and message column.
    """
    df = pd.read_csv(evidence_file, skiprows=1) # since the first row contains sep=,
    if 'CUSTOM.date [local]' in df.columns and 'CUSTOM.updateTime [local]' in df.columns and 'APP.tip' in df.columns:
        messages = df[['CUSTOM.date [local]', 'CUSTOM.updateTime [local]', 'APP.tip']]
        messages = messages.rename(columns={'CUSTOM.date [local]': 'date', 'CUSTOM.updateTime [local]': 'time', 'APP.tip': 'message'})
    elif 'CUSTOM.date [local]' in df.columns and 'CUSTOM.updateTime [local]' in df.columns and 'APP.warning' in df.columns:
        messages = df[['CUSTOM.date [local]', 'CUSTOM.updateTime [local]', 'APP.warning']]
        messages = messages.rename(columns={'CUSTOM.date [local]': 'date', 'CUSTOM.updateTime [local]': 'time', 'APP.warning': 'message'})
    elif 'CUSTOM.date [local]' in df.columns and 'CUSTOM.updateTime [local]' in df.columns and 'APP.error' in df.columns:
        messages = df[['CUSTOM.date [local]', 'CUSTOM.updateTime [local]', 'APP.error']]
        messages = messages.rename(columns={'CUSTOM.date [local]': 'date', 'CUSTOM.updateTime [local]': 'time', 'APP.error': 'message'})
    else:
        raise ValueError("The required columns are not present in the CSV file.")
    
    # Combine date and time into a single timestamp column
    messages['timestamp'] = pd.to_datetime(messages['date'] + ' ' + messages['time'], errors='coerce')
    # messages = messages.drop(columns=['date', 'time'])
    messages = messages.dropna(subset=['timestamp', 'message'])
    messages = messages.sort_values(by='timestamp').reset_index(drop=True)
    messages = messages.drop(columns=['timestamp'])
    
    return messages


def interpretability_report(model, tokenizer, config: dict, device, workdir: str, test_set: pd.DataFrame) -> pd.DataFrame:
    max_seq_length = 64
    print("Test set loaded successfully...")
    try:
        embedding_layer = get_embedding_layer(model.embedding_model)
        # print(f"Identified embedding layer: {embedding_layer}")
        lig = LayerIntegratedGradients(model, embedding_layer)
        print("LayerIntegratedGradients instantiated successfully!")
    except AttributeError as e:
        print(f"Error finding embedding layer: {e}")
        print("Please inspect your model structure carefully using print(your_model_instance)")
    print("Start interpreting...")
    vis_data_records_ig = []
    attribution_list = []
    pred_label_list = []
    pred_prob_list = []
    for index, row in test_set.iterrows():
        inputs = tokenizer(row['sentence'], return_tensors='pt', padding=True, truncation=True, max_length=max_seq_length)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        pred_label, pred_prob = infer_pred(model, input_ids, attention_mask, idx2pro)
        pred_label_list.append(pred_label)
        pred_prob_list.append(pred_prob)
        labelidx = pro2idx.get(pred_label, 0)
        attributions, delta = lig.attribute(inputs=input_ids, 
                                        baselines=input_ids*0, 
                                        additional_forward_args=(attention_mask,),
                                        target=labelidx,
                                        return_convergence_delta=True)
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        if config['embedding_model_name'] == 'modern-bert':
            tokens, attributions = reconstruct_roberta_tokens(tokens, attributions)
        else:      
            tokens, attributions = reconstruct_tokens(tokens, attributions)
        visualizer = add_attributions_to_visualizer(attributions, tokens, pred_prob, pred_label, None, delta)
        vis_data_records_ig.append(visualizer)
        attribution_list.append({
            "index": index + 1,
            "words": tokens,
            "attributions": attributions,
            "label": None,
            "pred_label": pred_label,
            "pred_prob": pred_prob,
        })
    html_output = visualization.visualize_text(vis_data_records_ig)
    with open(os.path.join(workdir, f'word_importance.html'), 'w') as f:
        f.write(html_output.data)
    with open(os.path.join(workdir, f"attributions.json"), "w", encoding="utf-8") as f:
        json.dump(attribution_list, f, indent=2, ensure_ascii=False, default=to_serializable)
    if name == 'nt':
        path_to_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    else:
        path_to_wkhtmltopdf = r'/usr/bin/wkhtmltopdf'
    config_wkhtml = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)
    pdfkit.from_string(html_output.data, os.path.join(workdir, f'word_importance.pdf'), configuration=config_wkhtml)
    print("Finish interpreting...")
    test_set['problem_type'] = pred_label_list
    test_set['pred_prob'] = pred_prob_list

    return test_set