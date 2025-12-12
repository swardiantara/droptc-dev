import os
import time
import json
import torch
import pandas as pd

from src.cli.preprocessing import segment_evidence
from src.cli.postprocessing import cluster_sentences, summarize_evidence, create_timeline_chart, create_message_timeline
from src.cli.utils import load_config, create_workdir, interpretability_report
from src.droptc.model import DroPTC

from transformers import AutoTokenizer, AutoModel


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Drone Log Analyzer")
    parser.add_argument("--config", default="src/cli/config.json", help="Path to the configuration file")
    parser.add_argument("--evidence_dir", default="src/evidence/decrypted", help="Path to the directory containing evidence files")
    return parser.parse_args()

def main():
    args = parse_args()
    timings = {
        'overall_start': time.time(),
        'config_load': None,
        'model_load': None,
        'files': {}
    }
    # load the config
    if args.config:
        t0 = time.time()
        config = load_config(args.config)
        timings['config_load'] = time.time() - t0
        if args.evidence_dir:
            evidence_dir = args.evidence_dir
        else:
            evidence_dir = config['source_evidence']
    else:
        t0 = time.time()
        config = load_config('src/cli/config.json')
        timings['config_load'] = time.time() - t0
        evidence_dir = config['source_evidence']
    workdir = create_workdir(config)
    
    # reading evidence files
    evidence_files = os.listdir(evidence_dir)
    evidence_files = [file for file in evidence_files if os.path.splitext(file)[1] == '.csv' ]
    print(f"Found {len(evidence_files)} evidence files in {evidence_dir}")
    
    # instantiate model
    #  Perform problem identification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(config['model_config']['embedding_model_name'], trust_remote_code=True)
    embedding_model = AutoModel.from_pretrained(config['model_config']['embedding_model_name'], trust_remote_code=True).to(device)
    model = DroPTC(embedding_model, tokenizer, freeze_embedding=True).to(device)
    model.load_state_dict(torch.load(config['model_dir'], map_location=device))
    model.eval()
    timings['model_load'] = time.time() - t0
    # process each evidence file
    for file in evidence_files:
        print(f"Processing {file}...")
        filename = os.path.splitext(file)[0]
        current_dir = os.path.join(workdir, filename)
        os.makedirs(current_dir, exist_ok=True)
        file_timing = {'start': time.time()}
        try:
            # Segment the evidence file and save the output to the current working directory
            t_step = time.time()
            current_evidence = segment_evidence(os.path.join(evidence_dir, file), current_dir)
            file_timing['segment'] = time.time() - t_step
            print(f"Segmented {len(current_evidence)} log messages from {file}")

            # Predict the problem type for each sentence in the segmented evidence
            t_step = time.time()
            current_evidence = interpretability_report(model, tokenizer, config['model_config'], device, current_dir, current_evidence)
            file_timing['interpretability'] = time.time() - t_step

            # Perform clustering, add cluster_id column to the parsed dataframe.
            t_step = time.time()
            current_evidence = cluster_sentences(current_evidence)
            file_timing['clustering'] = time.time() - t_step

            t_step = time.time()
            current_evidence.to_excel(os.path.join(current_dir, f'final.xlsx'), index=False)
            file_timing['save_final_excel'] = time.time() - t_step

            t_step = time.time()
            key_events = current_evidence[current_evidence['problem_type'] != 'Normal']
            key_events.to_excel(os.path.join(current_dir, f'key_events.xlsx'), index=False)
            file_timing['save_key_events_excel'] = time.time() - t_step
            print(f"Identified {len(key_events)} key events in {file}")

            # generate frequency report based on problem_type and cluster_id
            t_step = time.time()
            summarize_evidence(current_evidence, current_dir)
            file_timing['summarize'] = time.time() - t_step

            t_step = time.time()
            create_timeline_chart(current_evidence, current_dir)
            file_timing['create_timeline_chart'] = time.time() - t_step

            t_step = time.time()
            create_message_timeline(current_evidence, current_dir)
            file_timing['create_message_timeline'] = time.time() - t_step

            file_timing['end'] = time.time()
            file_timing['total'] = file_timing['end'] - file_timing['start']
            file_timing['n_messages'] = len(current_evidence)
            file_timing['n_key_events'] = len(key_events)
            timings['files'][file] = file_timing

            # write per-file timings for convenience
            with open(os.path.join(current_dir, 'timings.json'), 'w') as fh:
                json.dump(file_timing, fh, indent=2)

            print(f"Generated summary reports for {file}")
        except Exception as e:
            file_timing['error'] = str(e)
            file_timing['end'] = time.time()
            file_timing['total'] = file_timing['end'] - file_timing['start']
            timings['files'][file] = file_timing
            with open(os.path.join(current_dir, 'timings.json'), 'w') as fh:
                json.dump(file_timing, fh, indent=2)
            print(f"Error processing {file}: {e}")
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

    # finalize and write overall timings summary
    try:
        timings['overall_end'] = time.time()
        timings['overall_total'] = timings['overall_end'] - timings['overall_start']
        out_path = os.path.join(workdir, 'timings_summary.json')
        with open(out_path, 'w') as fh:
            json.dump(timings, fh, indent=2)
        print(f"Wrote overall timings to {out_path}")
    except Exception as e:
        print(f"Failed to write timings summary: {e}")

if __name__ == "__main__":
    main()