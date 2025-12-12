import json
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# 1. Add the file paths for each model's attribution data
ATTRIBUTION_FILES = {
    # 'MiniLM': os.path.join('experiments', 'droptc', 'sentence', 'all-MiniLM-L6-v2', 'unfreeze', '67351593', 'attributions_sentence.json'),
    # 'MiniLM*': os.path.join('experiments', 'droptc', 'sentence', 'DroPTC-all-MiniLM-L6-v2-sentence', 'unfreeze', '99511865', 'attributions_sentence.json'),
    # 'MPNet': os.path.join('experiments', 'droptc', 'sentence', 'all-mpnet-base-v2', 'unfreeze', '24677315', 'attributions_sentence.json'),
    # 'MPNet*': os.path.join('experiments', 'droptc', 'sentence', 'DroPTC-all-mpnet-base-v2-sentence', 'unfreeze', '87212562', 'attributions_sentence.json'),
    # 'BERT-base': os.path.join('experiments', 'droptc', 'sentence', 'bert-base-uncased', 'unfreeze', '52680723', 'attributions_sentence.json'),
    # 'NeoBERT': os.path.join('experiments', 'droptc', 'sentence', 'neo-bert', 'unfreeze', '14298463', 'attributions_sentence.json'),
    # 'ModernBERT': os.path.join('experiments', 'droptc', 'sentence', 'modern-bert', 'unfreeze', '87212562', 'attributions_sentence.json'),
    'DroPTC': os.path.join('experiments', 'droptc', 'sentence', 'DroPTC-all-mpnet-base-v2-sentence', 'unfreeze', 'ce-inverse', '37622020', 'attributions_sentence.json'),
    'DroPTC-WoCW': os.path.join('experiments', 'droptc', 'sentence', 'DroPTC-all-MiniLM-L6-v2-sentence', 'unfreeze', 'ce-uniformold', '87212562', 'attributions_sentence.json'),
    'DroneLog': os.path.join('experiments', 'dronelog', 'sentence', 'DroPTC-all-mpnet-base-v2-sentence', 'freeze', 'ce-uniform', '14298463', 'attributions_sentence.json'),
    'DroLoVe': os.path.join('experiments', 'drolove', 'sentence', 'bert-base-uncased', 'unfreeze', 'ce-uniform', '14298463', 'attributions_sentence.json'),
    'NeuralLog': os.path.join('experiments', 'neurallog', 'sentence', 'bert-base-uncased', 'unfreeze', 'ce-uniform', '70681460', 'attributions_sentence.json'),
    'TransSentLog': os.path.join('experiments', 'transsentlog', 'sentence', 'bert-base-uncased', 'unfreeze', 'ce-uniform', '14298463', 'attributions_sentence.json'),
}

OUTPUT_DIR = os.path.join('visualization', 'word-importance-compare')

def get_all_sample_indices(filepath: str) -> list:
    """Scans a file to find all unique sample indices."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return sorted([item.get("index") for item in data if "index" in item])
    except FileNotFoundError:
        return []

def load_sample_data(filepaths: dict, sample_index: int) -> tuple:
    """Loads the data for a specific sample index from multiple attribution files."""
    all_model_attrs = {}
    words, true_label = None, "Unknown"
    for model_name, path in filepaths.items():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            sample_data = next((item for item in data if item.get("index") == sample_index), None)
            if sample_data:
                if words is None:
                    words = sample_data['words']
                    true_label = sample_data.get('label', 'Unknown')
                if model_name == 'ModernBERT':
                    if len(sample_data['words']) == len(words): # if the ModernBERT tokenizes the input exactly the same as BERT-based embeddings
                        all_model_attrs[model_name] = sample_data['attributions']
                else:
                    all_model_attrs[model_name] = sample_data['attributions']
        except FileNotFoundError:
            continue # Skip if a file is missing
    return words, all_model_attrs, true_label

def normalize_for_diverging_heatmap(scores_dict: dict) -> np.ndarray:
    """
    Normalizes all scores by the max absolute value across all models
    for a single sample, preserving the [-1, 1] range.
    """
    # all_scores = np.concatenate(list(scores_dict.values()))
    # max_abs_val = np.max(np.abs(all_scores))
    # if max_abs_val == 0:
    #     max_abs_val = 1 # Avoid division by zero
    
    normalized_data = []
    for model_name in scores_dict.keys():
        normalized_data.append(scores_dict[model_name])
        
    return np.array(normalized_data)

def main():
    # --- Setup ---
    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
        
    # Get the list of all samples to process from the first model's file
    first_model_file = list(ATTRIBUTION_FILES.values())[0]
    sample_indices = get_all_sample_indices(first_model_file)
    
    if not sample_indices:
        print(f"Error: Could not find any samples in {first_model_file}. Exiting.")
        return

    print(f"Found {len(sample_indices)} samples. Starting heatmap generation...")

    # --- Main Loop ---
    for index in sample_indices:
        words, all_model_attrs, true_label = load_sample_data(ATTRIBUTION_FILES, index)
        
        if not words or not all_model_attrs:
            print(f"Skipping sample {index} due to missing data.")
            continue
            
        model_names = list(all_model_attrs.keys())
        heatmap_data = np.array([all_model_attrs[name] for name in model_names])
        # heatmap_data = normalize_for_diverging_heatmap(all_model_attrs)
        
        # --- Visualization ---
        # fig, ax = plt.subplots(figsize=(max(12, len(words) * 0.8), max(4, len(model_names) * 0.6)))
        fig, ax = plt.subplots(figsize=(15, 1.75))
        sns.heatmap(
            heatmap_data,
            xticklabels=words,
            yticklabels=model_names,
            cmap="Greens", # Diverging colormap (Red-White-Blue)
            annot=True,   # Show numeric values
            fmt=".2f",
            annot_kws={"size": 8.5},
            cbar_kws={
                'shrink': 0.9,     # shrink colorbar height (tune this value)
                'aspect': 10,       # controls colorbar thickness (higher = thinner)
                'pad': 0.005   # smaller pad = closer to the heatmap
            },
            linewidths=.5,
            center=0,     # Anchor the colormap at zero
            vmin=-1,      # Set explicit limits
            vmax=1,
            ax=ax
        )
        
        # ax.set_title(f"Attribution Comparison for Sample {index} (Label: {true_label})", fontsize=16)
        # ax.set_xlabel("Log Sentence", fontsize=10)
        ax.set_ylabel("")
        plt.xticks()
        plt.yticks()
        
        # --- Save File ---
        filename = f"sample_{index}_{true_label}.pdf"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close(fig) # Close the figure to free up memory
        
    print(f"\nProcessing complete. All heatmaps saved in the '{OUTPUT_DIR}' folder.")

if __name__ == '__main__':
    main()