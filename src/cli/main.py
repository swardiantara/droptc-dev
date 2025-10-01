import os
import torch
import pandas as pd

from src.cli.preprocessing import segment_evidence
from src.cli.postprocessing import cluster_sentences, summarize_evidence
from src.cli.utils import load_config, create_workdir, interpretability_report
from src.droptc.model import DroPTC

from transformers import AutoTokenizer, AutoModel


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Drone Log Analyzer")
    parser.add_argument("--config", default="src/cli/config.json", help="Path to the configuration file")
    parser.add_argument("--evidence_dir", default=None, help="Path to the directory containing evidence files")
    return parser.parse_args()

def main():
    args = parse_args()
    # load the config
    if args.config:
        config = load_config(args.config)
        if args.evidence_dir:
            evidence_dir = args.evidence_dir
        else:
            evidence_dir = config['source_evidence']
    else:
        config = load_config('src/cli/config.json')
        evidence_dir = config['source_evidence']
    workdir = create_workdir(config)
    
    # reading evidence files
    evidence_files = os.listdir(evidence_dir)
    evidence_files = [file for file in evidence_files if os.path.splitext(file)[1] == '.csv' ]
    print(f"Found {len(evidence_files)} evidence files in {evidence_dir}")
    
    # instantiate model
    #  Perform problem identification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config['model_config']['embedding_model_name'], trust_remote_code=True)
    embedding_model = AutoModel.from_pretrained(config['model_config']['embedding_model_name'], trust_remote_code=True).to(device)
    model = DroPTC(embedding_model, tokenizer, freeze_embedding=True).to(device)
    model.load_state_dict(torch.load(config['model_dir'], map_location=device))
    model.eval()
    # process each evidence file
    for file in evidence_files:
        print(f"Processing {file}...")
        filename = os.path.splitext(file)[0]
        current_dir = os.path.join(workdir, filename)
        os.makedirs(current_dir, exist_ok=True)

        # Segment the evidence file and save the output to the current working directory
        current_evidence = segment_evidence(os.path.join(evidence_dir, file), current_dir)
        print(f"Segmented {len(current_evidence)} log messages from {file}")
        # Predict the problem type for each sentence in the segmented evidence
        current_evidence = interpretability_report(model, tokenizer, config['model_config'], device, current_dir, current_evidence)

        # Perform clustering, add cluster_id column to the parsed dataframe.
        current_evidence = cluster_sentences(current_evidence)
        current_evidence.to_excel(os.path.join(current_dir, f'final.xlsx'), index=False)
        # generate frequency report based on problem_type and cluster_id
        summarize_evidence(current_evidence, current_dir)
    # Read raw decrypted evidence file, extract message column, store in dataframe variable
    # Export the parsed log into an excel file, store it under the output folder -> parsed_XXXfilename
    # Perform log segmentation, add message_id and sentence columns to the parsed dataframe
    # Perform problem identification, add problem_type column to the parsed dataframe
        # During inference, compute feature attribution towards the predicted label, save the attribution score and generate feature attribution report as well.
    # Perform clustering, add cluster_id column to the parsed dataframe.
    # Export the dataframe to excel and json file
    # Generate evidence summary report -> event frequency table
    # Generate timeline summary report -> gantt-style chart
    # Generate forensic timeline -> highlight given to problem-indicating logs
    # Generate forensic report -> investigation data, evidence, summary, and timeline. -> PDF File

if __name__ == "__main__":
    main()