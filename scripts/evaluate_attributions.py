import os
import json
import pandas as pd
import string

def clean_token(token):
    """Removes special characters and checks if token is valid content."""
    # List of artifacts to ignore
    ignore_list = ['<s>', '</s>', '[CLS]', '[SEP]', '[PAD]']
    
    # Check if it's a special token
    if token in ignore_list:
        return False, token
    
    # Check if it's just punctuation (e.g., ".", ",")
    if token.strip() in string.punctuation:
        return False, token
        
    return True, token

def evaluate_interpretability(annotation_file, attribution_json_file, top_k=3):
    # 1. Load Data
    annotated_df = pd.read_excel(annotation_file)
    with open(attribution_json_file, 'r') as f:
        attributions_data = json.load(f)
    
    # Convert JSON to a dict for fast lookup by index (assuming 'index' matches 'original_index')
    # If indices don't match, we can match by sentence text.
    attr_dict = {item['index']: item for item in attributions_data}
    
    scores = []
    
    print(f"{'Index':<6} | {'Label':<25} | {'Prec@k':<8} | {'Match Details'}")
    print("-" * 80)
    word_count = 0
    match_count = 0
    dataframe = []
    # k, index, precision
    for _, row in annotated_df.iterrows():
        idx = row['index']
        ground_truth_text = str(row['relevant_words'])
        
        # Skip if annotation is empty
        if pd.isna(ground_truth_text) or ground_truth_text.strip() == "":
            continue
            
        # Parse Human Annotation
        human_keywords = [w.strip().lower() for w in ground_truth_text.split(',')]
        
        # Retrieve Model Data
        if idx not in attr_dict:
            # Fallback: Try matching by text if index matching fails
            # (Optional implementation, skipping for now)
            print(f"Warning: Index {idx} not found in JSON.")
            continue
            
        entry = attr_dict[idx]
        tokens = entry['words']
        attr_vals = entry['attributions']
        
        # 2. Preprocess Model Tokens
        # Zip tokens with scores and filter out special tokens/punctuation
        valid_pairs = []
        for t, s in zip(tokens, attr_vals):
            is_valid, cleaned_t = clean_token(t)
            if is_valid:
                valid_pairs.append((cleaned_t, s))
        
        # 3. Select Top-k Model Tokens
        # We sort by Value (Descending) because we want the features that *positively* contributed
        # to the predicted class.
        valid_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract just the words
        model_ranked_words = [t[0].lower() for t in valid_pairs]
        
        # 4. Calculate Adaptive k
        # k = min(3, length of valid sentence)
        k = min(top_k, len(model_ranked_words))
        
        if k == 0:
            scores.append(0)
            continue
            
        top_k_model_words = model_ranked_words[:k]
        
        # 5. Calculate Precision
        # Count how many of the top_k model words are in the human list
        matches = sum(1 for word in top_k_model_words if word in human_keywords)
        precision = matches / k
        match_count += matches
        word_count += k
        scores.append(precision)
        dataframe.append({
            'index': idx,
            'precision': precision
            })
        # Debug Print (Optional: Show matches for first few)
        print(f"{idx:<6} | {row['problem_type']:<25} | {precision:.2f}     | {top_k_model_words} vs {human_keywords}")

    # Final Result
    precision = match_count / word_count if word_count > 0 else 0
    avg_precision = sum(scores) / len(scores) if scores else 0
    print("-" * 80)
    print(f"Evaluated {len(scores)} samples.")
    print(f"Total Matches: {match_count} out of {word_count}, precision: {precision:.4f}")
    print(f"Average Adaptive Precision@k: {avg_precision:.4f}")

    return precision, avg_precision, dataframe


if __name__ == "__main__":
    # --- Run the evaluation ---
    # Out of 10 runs, we choose the best-performing models based on f1 score
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
    k_values = [1, 2, 3]
    reports = []
    complete_results = pd.DataFrame()
    for model_name, path in ATTRIBUTION_FILES.items():
        for k in k_values:
        # attribution_file = os.path.join('attributions', f'attributions_{model_config}.json')
            print(f"\nEvaluating model: {model_name}")
            precision, avg_precision, dataframe = evaluate_interpretability(os.path.join('dataset', 'test_problem_sentence.xlsx'), path, k)
            reports.append({
                'model_name': model_name,
                'top_k': k,
                'precision': precision,
                'avg_precision': avg_precision})
            temp_df = pd.DataFrame(dataframe)
            temp_df['model_name'] = model_name
            temp_df['top_k'] = k
            complete_results = pd.concat([complete_results, temp_df], ignore_index=True)
    results_df = pd.DataFrame.from_dict(reports).reset_index().rename(columns={'index': 'model_name'}   )
    results_df = results_df.sort_values(by='avg_precision', ascending=False)
    results_df.to_excel(os.path.join('experiments', 'analysis', 'interpretability_evaluation_results.xlsx'), index=False)
    complete_results[['model_name', 'top_k', 'index', 'precision']].sort_values(by=['model_name', 'top_k', 'index']).to_excel(os.path.join('experiments', 'analysis', 'interpretability_detailed_results.xlsx'), index=False)
    # evaluate_interpretability('samples_to_annotate.xlsx', 'attributions_sentence.json')