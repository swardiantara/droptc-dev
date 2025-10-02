import os
import time
import threading
import pandas as pd
import torch
import psutil
import argparse
import json
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from src.droptc.model import DroPTC
from src.droptc.utils import SentenceDataset
from src.droptc.train_classifier import slabel2idx

# Helper class to monitor system resources in a separate thread
class SystemMonitor(threading.Thread):
    def __init__(self, device='cpu', gpu_index=0):
        super().__init__()
        self.device = device
        self.gpu_index = gpu_index
        self.stopped = threading.Event()
        self.cpu_usage = []
        self.gpu_usage = []
        self.daemon = True
        if self.device == 'cuda' and NVML_AVAILABLE:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

    def run(self):
        while not self.stopped.is_set():
            # Record CPU usage
            self.cpu_usage.append(psutil.cpu_percent())
            
            # Record GPU usage if applicable
            if self.device == 'cuda' and NVML_AVAILABLE:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    self.gpu_usage.append(util.gpu)
                except pynvml.NVMLError:
                    # Handle cases where the GPU might not be available or query fails
                    self.gpu_usage.append(0)
            
            time.sleep(0.1) # Poll every 100ms

    def stop(self):
        self.stopped.set()
        if self.device == 'cuda' and NVML_AVAILABLE:
            pynvml.nvmlShutdown()

    def get_results(self):
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        peak_cpu = max(self.cpu_usage) if self.cpu_usage else 0
        avg_gpu = sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else 0
        peak_gpu = max(self.gpu_usage) if self.gpu_usage else 0
        return avg_cpu, peak_cpu, avg_gpu, peak_gpu

def get_prediction_pipeline(model_name, device):
    """Loads the model and tokenizer and returns a prediction function."""
    # Load embedding model
    tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{model_name}")
    embedding_model = AutoModel.from_pretrained(f"sentence-transformers/{model_name}").to(device)
    embedding_model.eval()

    # Load classifier
    # Assuming the classifier is compatible with both embedding models
    classifier = DroPTC(embedding_model, tokenizer, freeze_embedding=True).to(device)
    
    # Path to the trained classifier model
    # IMPORTANT: This path might need to be adjusted based on your project structure
    if model_name == 'all-MiniLM-L6-v2':
        classifier_path = 'src/cli/model/pytorch_model.pt'
    else:
        classifier_path = 'src/cli/model/pytorch_model_mpnet.pt'
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(f"Classifier model not found at {classifier_path}. Please ensure the path is correct.")
        
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    classifier.eval()

    def predict(test_loader):
        """Function to run the full prediction pipeline."""
        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = classifier(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
        
        return all_preds

    return predict

def run_single_test(model_name, device, data_sample):
    """
    Runs a single efficiency test for a given configuration.
    Includes a warm-up run to ensure fair measurement.
    """
    print(f"  Testing model: {model_name} on {device.upper()} with {len(data_sample)} samples...")

    # 1. Load model and create prediction pipeline
    try:
        predict_pipeline = get_prediction_pipeline(model_name, device)
    except Exception as e:
        print(f"    ERROR: Failed to load model. {e}")
        return None

    # 2. Warm-up run (not measured)
    # This helps to cache models and initialize CUDA contexts to avoid one-off startup costs
    try:
        print("    Performing warm-up run...")
        warmup_sample = iter(data_sample)
        if warmup_sample:
            predict_pipeline(next(warmup_sample))
        print("    Warm-up complete.")
    except Exception as e:
        print(f"    ERROR during warm-up: {e}")
        return None

    # 3. Measured run
    monitor = SystemMonitor(device=device)
    
    # The data_sample is already a list of sentences
    # sentences_to_predict = data_sample

    if not data_sample:
        print("    No sentences provided. Skipping test.")
        return None

    torch.cuda.synchronize() if device == 'cuda' else None
    
    monitor.start()
    start_time = time.perf_counter()

    # Execute the pipeline
    predict_pipeline(data_sample)

    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.perf_counter()
    monitor.stop()
    
    inference_time = end_time - start_time
    avg_cpu, peak_cpu, avg_gpu, peak_gpu = monitor.get_results()

    print(f"    Inference Time: {inference_time:.4f}s")
    print(f"    Avg/Peak CPU: {avg_cpu:.2f}% / {peak_cpu:.2f}%")
    if device == 'cuda':
        print(f"    Avg/Peak GPU: {avg_gpu:.2f}% / {peak_gpu:.2f}%")

    return {
        "model": model_name.split('/')[-1],
        "device": device,
        "sample_size": len(data_sample),
        "inference_time_s": inference_time,
        "avg_cpu_percent": avg_cpu,
        "peak_cpu_percent": peak_cpu,
        "avg_gpu_percent": avg_gpu,
        "peak_gpu_percent": peak_gpu,
    }


def resample_dataframe(df, target_size, random_state=42):
    """
    Resample a dataframe to a target size by duplicating or selecting rows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe to resample
    target_size : int
        Desired number of samples in the output dataframe
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    pd.DataFrame
        Resampled dataframe with target_size rows
    """
    current_size = len(df)
    
    if target_size <= current_size:
        # Downsample: randomly select rows
        return df.sample(n=target_size, random_state=random_state).reset_index(drop=True)
    else:
        # Upsample: duplicate rows
        # Calculate how many full copies and remaining samples needed
        n_full_copies = target_size // current_size
        n_remaining = target_size % current_size
        
        # Create list of dataframes to concatenate
        dfs = [df] * n_full_copies
        
        # Add remaining samples if needed
        if n_remaining > 0:
            dfs.append(df.sample(n=n_remaining, random_state=random_state))
        
        # Concatenate and reset index
        return pd.concat(dfs, ignore_index=True)


def run_efficiency_test(args, workdir: str):
    """
    Main function to orchestrate the efficiency testing across different models,
    devices, and sample sizes.
    """
    # --- Configuration ---
    # MODELS_TO_TEST = [
    #     'sentence-transformers/all-MiniLM-L6-v2', # Proposed model
    #     'sentence-transformers/all-mpnet-base-v2'   # Baseline model
    # ]
    # DEVICES_TO_TEST = ['cpu']
    # if torch.cuda.is_available():
    #     DEVICES_TO_TEST.append('cuda')
    
    DATASET_PATH = 'dataset/test_sentence.xlsx'
    # SAMPLE_SIZES = [1000, 5000, 10000, 50000, 100000, 500000, 1000000] # Add more sizes if needed

    # --- Test Execution ---
    print("Starting Efficiency Test...")
    
    # Load data
    try:
        full_df = pd.read_excel(DATASET_PATH)
        full_df['label'] = full_df['problem_type'].map(slabel2idx)
        # The pipeline expects a 'sentence' column.
        if 'sentence' not in full_df.columns:
            raise ValueError("Dataset must contain a 'sentence' column.")
        
        print(f"Loaded dataset with {len(full_df)} records.")
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {DATASET_PATH}. Aborting.")
        return

    all_results = []

    # for device in DEVICES_TO_TEST:
    #     print(f"\n----- Testing on Device: {device.upper()} -----")
    #     for model_name in MODELS_TO_TEST:
    #         workdir = os.path.join(output_path, model_name.split('/')[-1], device)
            
    print(f"\n  Preparing sample size: {args.sample_size}")
    # Resample the dataframe to the target args.sample_size
    resampled_df = resample_dataframe(full_df, args.sample_size)
    # data loader sini. ganti sample_data jadi data loader
    tokenizer = AutoTokenizer.from_pretrained(f"sentence-transformers/{args.model_name}")
    dataset = SentenceDataset(resampled_df, tokenizer, max_length=64)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    result = run_single_test(args.model_name, args.device, data_loader)

    if result:
        all_results.append(result)
        with open(os.path.join(workdir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=4) # Indents with 4 spaces

    # --- Reporting ---
    # if not all_results:
    #     print("\nNo results were generated. Test finished.")
    #     return
        
    # results_df = pd.DataFrame(all_results)
    
    
    # results_df.to_csv(os.path.join(workdir, 'efficiency_test_results.csv'), index=False)

    print(f"\nEfficiency test complete. Results saved to '{workdir}'")
    # print(results_df.round(4))



def main():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument('--sample_size', type=int, default=1000, help='Number of samples')
    parser.add_argument('--model_name', type=str, choices=['all-MiniLM-L6-v2', 'all-mpnet-base-v2'], default='all-MiniLM-L6-v2', help='Type of Word Embdding used. Default: `all-MiniLM-L6-v2`')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                    help="Device to perform the computation. Default: `cpu`.")

    args = parser.parse_args()
    output_path = os.path.join('experiments', 'efficiency_test', args.model_name, args.device, str(args.sample_size))
    os.makedirs(output_path, exist_ok=True)
    if os.path.exists(os.path.join(output_path, 'result.json')):
        print('Scenario has been executed. Skipped!')
        return exit(0)
    run_efficiency_test(args, output_path)

if __name__ == '__main__':
    main()
