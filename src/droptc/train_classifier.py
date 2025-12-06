import os
import json
import copy
import pdfkit
import random
import argparse
import torch

import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os import name
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, f1_score
from captum.attr import LayerIntegratedGradients, visualization

from src.droptc.utils import SentenceDataset, visualize_sentence, interactive_plot
from src.droptc.interpretability import get_embedding_layer, infer_pred, reconstruct_roberta_tokens, reconstruct_tokens, add_attributions_to_visualizer, to_serializable
from src.droptc.model import DroPTC, DroneLog, NeuralLog, TransSentLog, DroLoVe

raw2pro = {
    'normal': 'Normal',
    'SurEnv': 'SurroundingEnvironment',
    'HwFlt': 'HardwareFault',
    'ConfIss': 'ParamViolation',
    'VioReg': 'RegulationViolation',
    'CommIss': "CommunicationIssue",
    'Swflt': 'SoftwareFault',
}

idx2pro = {
    0: 'Normal',
    1: 'SurroundingEnvironment',
    2: 'HardwareFault',
    3: 'ParamViolation',
    4: 'RegulationViolation',
    5: "CommunicationIssue",
    6: 'SoftwareFault'
}

pro2idx = {
    'Normal': 0,
    'SurroundingEnvironment': 1,
    'HardwareFault': 2,
    'ParamViolation': 3,
    'RegulationViolation': 4,
    'CommunicationIssue': 5,
    'SoftwareFault': 6
}

slabel2idx = {
    'normal': 0,
    'SurEnv': 1,
    'HwFlt': 2,
    'ConfIss': 3,
    'VioReg': 4,
    'CommIss': 5,
    'Swflt': 6,
}

sidx2label = {
    0: 'normal',
    1: 'SurEnv',
    2: 'HwFlt',
    3: 'ConfIss',
    4: 'VioReg',
    5: 'CommIss',
    6: 'Swflt'
}

MODEL_REGISTRY = {
    "droptc": DroPTC,
    "drolove": DroLoVe,
    "dronelog": DroneLog,
    "neurallog": NeuralLog,
    "transsentlog": TransSentLog,
}

def get_args():
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument("--feature_col", default="sentence", help="Level of analysis")
    parser.add_argument("--scenario", type=str, choices=['droptc', 'drolove', 'dronelog', 'neurallog', 'transsentlog'], default="droptc", help="The model used. Default: `droptc`")
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help="Folder to store the experimental results. Default: experiments")
    parser.add_argument('--embedding', type=str, choices=['bert-base-uncased', 'neo-bert', 'modern-bert', 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'DroPTC-all-mpnet-base-v2-sentence', 'DroPTC-all-MiniLM-L6-v2-sentence'], default='bert-base-uncased', help='Type of Word Embdding used. Default: `bert-base-uncased`')
    parser.add_argument('--encoder', type=str, choices=['transformer', 'lstm', 'gru', 'linear'], default='linear',
                    help="Encoder architecture used to perform computation. Default: `linear`.")
    parser.add_argument('--dim_reduce', type=str, choices=['tsne', 'pca-tsne', 'umap', 'pca', 'isomap'], default='pca-tsne', help='Dimensional reduction method for data visualization. Default: `pca-tsne`')
    parser.add_argument('--loss_fn', type=str, choices=['ce', 'focal'], default='ce',
                    help="Loss function to use for training. Default: `ce` (CrossEntropyLoss)")
    parser.add_argument('--class_weight', choices=['uniform', 'balanced', 'inverse'], default='uniform',
                    help="Wether to weigh the class based on the class frequency. Default: Uniform")
    parser.add_argument('--bidirectional', action='store_true',
                    help="Wether to use Bidirectionality for LSTM and GRU.")
    parser.add_argument('--n_heads', type=int, default=1,
                        help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of encoder layers')
    parser.add_argument('--n_epochs', type=int, default=15,
                        help='Number of testtraining iterations')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of samples in a batch')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--freeze_embedding', action='store_true',
                        help="Wether to freeze the pre-trained embedding's parameter.")
    parser.add_argument('--save_model', action='store_true',
                        help="Wether to save model.")
    parser.add_argument('--overwrite', action='store_true',
                        help="Wether to overwrite the previous run.")
    
    args = parser.parse_args()

    return args


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")


class FocalLoss(torch.nn.Module):
    """Focal Loss implementation for handling class imbalance.
    
    Uses class weights as alpha parameters for each class instead of a constant alpha.
    Gamma is fixed to 2.0 (standard value).
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight  # Class weights used as alpha parameters
        self.gamma = gamma  # Fixed at 2.0
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(labels, weight_type='uniform', num_classes=None, class_map=None):
    """Compute class weights for handling class imbalance.

    This function is careful to align weights with class indices. If you use a
    custom mapping (e.g. `class_name -> index`) pass it via `class_map` so the
    returned tensor matches your model's expected class ordering.

    Args:
        labels: iterable of labels (either integer indices or class names)
        weight_type: 'uniform', 'balanced', or 'inverse'
        num_classes: number of classes (optional if `class_map` provided)
        class_map: optional dict mapping class_name -> class_index

    Returns:
        torch.Tensor of class weights (length == num_classes) or None
    """
    if weight_type == 'uniform':
        return None

    # If class_map is provided, determine num_classes from it
    if class_map is not None:
        num_classes = len(class_map)

    # If still unknown, try to infer from labels
    if num_classes is None:
        try:
            labels_array = np.array(labels)
            max_label = labels_array.max()
            num_classes = int(max_label) + 1
        except Exception:
            raise ValueError("num_classes could not be inferred; provide num_classes or class_map")

    # Initialize counts aligned to indices 0..num_classes-1
    class_counts = np.zeros(num_classes, dtype=np.int64)

    if class_map is not None:
        # labels are assumed to be class names; map to indices
        for lab in labels:
            if lab in class_map:
                idx = class_map[lab]
                if 0 <= idx < num_classes:
                    class_counts[idx] += 1
            else:
                # If label not in map, try converting to int
                try:
                    idx = int(lab)
                    if 0 <= idx < num_classes:
                        class_counts[idx] += 1
                except Exception:
                    # ignore unknown labels
                    pass
    else:
        # labels should be integer indices
        labels_array = np.array(labels)
        # Clip/convert labels that are out-of-range
        for lab in labels_array:
            try:
                idx = int(lab)
                if 0 <= idx < num_classes:
                    class_counts[idx] += 1
            except Exception:
                # ignore non-integer labels when no class_map provided
                pass

    # Avoid zero counts causing divide-by-zero
    counts = class_counts.astype(float)
    counts[counts == 0] = 0.0

    if weight_type == 'balanced':
        # Weight inversely proportional to class frequency
        # Small epsilon to avoid division by zero
        eps = 1e-8
        inv = 1.0 / (counts + eps)
        weights = inv / inv.sum() * num_classes
    elif weight_type == 'inverse':
        total = counts.sum()
        eps = 1e-8
        weights = total / (num_classes * (counts + eps))
    else:
        return None

    return torch.FloatTensor(weights)


def get_loss_fn(loss_type='ce', class_weights=None):
    """Factory function to return the appropriate loss function.
    
    Args:
        loss_type: 'ce' or 'focal'
        class_weights: torch.Tensor of class weights or None
    
    Returns:
        loss function instance
    """
    if loss_type == 'focal':
        return FocalLoss(weight=class_weights, gamma=2.0, reduction='mean')
    else:  # 'ce'
        return torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')



# --- Modularized Pipeline Functions ---
def prepare_data(args):
    """Load datasets and encode labels using sklearn's LabelEncoder.

    Assumes `problem_type` column contains the long class names (e.g.
    'SurroundingEnvironment', 'RegulationViolation'). Fits the encoder on
    the training set and transforms both train and test so indices are
    consistent. Returns (train_df, test_df, label_encoder).
    """
    train_df = pd.read_excel(os.path.join('dataset', f'train_{args.feature_col}.xlsx'))
    test_df = pd.read_excel(os.path.join('dataset', f'test_{args.feature_col}.xlsx'))

    # Expect long class names in problem_type; fit encoder on training set
    le = LabelEncoder()
    # Drop NA and ensure strings
    train_labels = train_df['problem_type'].astype(str).fillna('Unknown').tolist()
    le.fit(train_labels)

    # Transform and store both index and original name
    train_df['label'] = le.transform(train_df['problem_type'].astype(str).fillna('Unknown'))
    train_df['label_name'] = train_df['problem_type'].astype(str).fillna('Unknown')

    test_df['label'] = le.transform(test_df['problem_type'].astype(str).fillna('Unknown'))
    test_df['label_name'] = test_df['problem_type'].astype(str).fillna('Unknown')

    return train_df, test_df, le

def get_model_and_tokenizer(args, device):
    if str(args.embedding).startswith('DroPTC'):
        model_name_path = f"swardiantara/{args.embedding}"
    elif args.embedding == 'neo-bert':
        model_name_path = 'chandar-lab/NeoBERT'
    elif args.embedding == 'modern-bert':
        model_name_path = 'answerdotai/ModernBERT-base'
    elif str(args.embedding).startswith('all'):
        model_name_path = f'sentence-transformers/{args.embedding}'
    else:
        model_name_path = args.embedding
    tokenizer = AutoTokenizer.from_pretrained(model_name_path, trust_remote_code=True)
    embedding_model = AutoModel.from_pretrained(model_name_path, trust_remote_code=True).to(device)
    return embedding_model, tokenizer

def get_dataloaders(train_df, test_df, tokenizer, args):
    max_seq_length = 64
    batch_size = args.batch_size
    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    merged_dataset = SentenceDataset(merged_df, tokenizer, max_seq_length)
    merged_loader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=False)
    train_dataset = SentenceDataset(train_df, tokenizer, max_seq_length)
    test_dataset = SentenceDataset(test_df, tokenizer, max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return merged_df, merged_loader, train_loader, test_loader

def train(model, train_loader, test_loader, train_df, args, device, class_map=None):
    # Compute class weights based on training data (use human-readable names so
    # mapping from class_map -> indices is unambiguous)
    class_weights = compute_class_weights(train_df['label_name'].tolist(), args.class_weight, class_map=class_map)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    # Get the appropriate loss function
    criterion = get_loss_fn(args.loss_fn, class_weights)
    if isinstance(criterion, FocalLoss):
        criterion = criterion.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_acc_epoch = float('-inf')
    best_f1_epoch = float('-inf')
    best_epoch = 0
    best_model_state = None
    num_epochs = args.n_epochs
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_loss_epoch = total_train_loss / len(train_loader)
        print(f"{epoch+1}/{num_epochs}: train_loss: {train_loss_epoch}/{total_train_loss}")
        # Validation
        model.eval()
        val_epoch_labels = []
        val_epoch_preds = []
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                pred_probs = torch.softmax(logits, axis=1)
                pred_label = torch.argmax(pred_probs, axis=1).cpu().numpy()
                val_epoch_labels.extend(labels.cpu().numpy())
                val_epoch_preds.extend(pred_label)
            val_loss_epoch = total_val_loss / len(test_loader)
        print(f"{epoch+1}/{num_epochs}: val_loss: {val_loss_epoch}/{total_val_loss}")
        val_acc_epoch = accuracy_score(val_epoch_labels, val_epoch_preds)
        precision, recall, val_f1, _ = precision_recall_fscore_support(val_epoch_labels, val_epoch_preds, average='weighted')
        if (val_f1 > best_f1_epoch and val_acc_epoch > best_acc_epoch) or (val_f1 > best_f1_epoch and val_acc_epoch >= best_acc_epoch) or (val_f1 >= best_f1_epoch and val_acc_epoch > best_acc_epoch):
            best_f1_epoch = val_f1
            best_acc_epoch = val_acc_epoch
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
    return best_model_state, best_epoch, best_f1_epoch, best_acc_epoch

def evaluate(model, test_loader, test_df, args, device):
    model.eval()
    all_labels_multiclass = []
    all_preds_multiclass = []
    all_preds_probs_multiclass = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            logits_multiclass_test = model(input_ids, attention_mask)
            logits_multiclass_test = torch.softmax(logits_multiclass_test, dim=1)
            predicted_probs_multiclass_test, predicted_labels_multiclass_test = torch.max(logits_multiclass_test, dim=1)
            all_labels_multiclass.extend(labels.cpu().numpy())
            all_preds_multiclass.extend(predicted_labels_multiclass_test.cpu().numpy())
            all_preds_probs_multiclass.extend(predicted_probs_multiclass_test.cpu().numpy())
    preds_decoded = [idx2pro.get(key) for key in all_preds_multiclass]
    tests_decoded = [idx2pro.get(key) for key in all_labels_multiclass]
    prediction_df = pd.DataFrame()
    prediction_df["message"] = test_df["message"]
    prediction_df["sentence"] = test_df["sentence"]
    prediction_df["label"] = list(tests_decoded)
    prediction_df["pred"] = list(preds_decoded)
    prediction_df["verdict"] = [label == pred for label, pred in zip(tests_decoded, preds_decoded)]
    prediction_df["prob"] = all_preds_probs_multiclass
    return prediction_df, preds_decoded, tests_decoded

def report_results(prediction_df, preds_decoded, tests_decoded, workdir, args, best_epoch, best_f1_epoch, best_acc_epoch):
    prediction_df.to_excel(os.path.join(workdir, "prediction.xlsx"), index=False)
    accuracy = accuracy_score(tests_decoded, preds_decoded)
    f1_weighted = f1_score(tests_decoded, preds_decoded, average='weighted')
    evaluation_report = classification_report(tests_decoded, preds_decoded, digits=5)
    classification_report_result = classification_report(tests_decoded, preds_decoded, digits=5, output_dict=True)
    classification_report_result['macro_avg'] = classification_report_result.pop('macro avg')
    classification_report_result['weighted_avg'] = classification_report_result.pop('weighted avg')
    micro_pre, micro_rec, micro_f1, _ = precision_recall_fscore_support(tests_decoded, preds_decoded, average='micro')
    classification_report_result['micro_avg'] = {
        "precision": micro_pre,
        "recall": micro_rec,
        "f1-score": micro_f1
    }
    with open(os.path.join(workdir, "evaluation_report.json"), 'w') as json_file:
        json.dump(classification_report_result, json_file, indent=4)
    with open(os.path.join(workdir, "evaluation_report.txt"), "w") as text_file:
        text_file.write(evaluation_report)
    print("Best epoch: ", best_epoch)
    print("Accuracy:", accuracy)
    print("F1-score:", f1_weighted)
    print("Classification Report:\n", evaluation_report)
    arguments_dict = vars(args)
    arguments_dict['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    arguments_dict['scenario_dir'] = workdir
    arguments_dict['best_epoch'] = best_epoch
    arguments_dict['best_val_f1'] = best_f1_epoch
    arguments_dict['best_val_acc'] = best_acc_epoch
    # If a LabelEncoder was attached to args (added in main), include its classes
    if hasattr(args, 'label_classes'):
        arguments_dict['label_classes'] = args.label_classes
    with open(os.path.join(workdir, 'scenario_arguments.json'), 'w') as json_file:
        json.dump(arguments_dict, json_file, indent=4)
    # Confusion matrix: prefer classes from args if present, else fallback
    if hasattr(args, 'label_classes') and args.label_classes:
        class_names = args.label_classes
    else:
        class_names = [value for _, value in raw2pro.items()]
    conf_matrix = confusion_matrix(prediction_df['label'].to_list(), prediction_df['pred'].to_list(), labels=class_names)
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(conf_matrix, annot=True, xticklabels=class_names, yticklabels=class_names, fmt='d', cmap='YlGnBu', cbar=False, square=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(os.path.join(workdir, "confusion_matrix.pdf"), bbox_inches='tight')
    plt.close()

def visualize_and_save(merged_loader, idx2pro, best_model, device, workdir, merged_df, args, best_model_state):
    visualize_sentence(merged_loader, idx2pro, best_model.to(device), device, workdir)
    interactive_plot(merged_df, merged_loader, args.feature_col, 'label_name', best_model, device, args.dim_reduce, args.seed, workdir)
    if args.save_model:
        torch.save(best_model_state, os.path.join(workdir, 'sentence_pytorch_model.pt'))

def interpretability_report(model, tokenizer, args, device, workdir):
    max_seq_length = 64
    test_set = pd.read_excel(os.path.join('dataset', f'test_sentence.xlsx'))
    # Expect problem_type to contain long class names; use them directly
    test_set["label"] = test_set['problem_type'].astype(str).fillna('Unknown')
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
    for index, row in test_set.iterrows():
        label = row['label']
        labelidx = pro2idx.get(label)
        inputs = tokenizer(row[args.feature_col], return_tensors='pt', padding=True, truncation=True, max_length=max_seq_length)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        pred_label, pred_prob = infer_pred(model, input_ids, attention_mask, idx2pro)
        attributions, delta = lig.attribute(inputs=input_ids, 
                                        baselines=input_ids*0, 
                                        additional_forward_args=(attention_mask,),
                                        target=labelidx,
                                        return_convergence_delta=True)
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        if args.embedding == 'modern-bert':
            tokens, attributions = reconstruct_roberta_tokens(tokens, attributions)
        else:      
            tokens, attributions = reconstruct_tokens(tokens, attributions)
        visualizer = add_attributions_to_visualizer(attributions, tokens, pred_prob, pred_label, label, delta)
        vis_data_records_ig.append(visualizer)
        attribution_list.append({
            "index": index + 1,
            "words": tokens,
            "attributions": attributions,
            "label": label,
            "pred_label": pred_label,
            "pred_prob": pred_prob,
        })
    html_output = visualization.visualize_text(vis_data_records_ig)
    with open(os.path.join(workdir, f'word_importance_{args.feature_col}.html'), 'w') as f:
        f.write(html_output.data)
    with open(os.path.join(workdir, f"attributions_{args.feature_col}.json"), "w", encoding="utf-8") as f:
        json.dump(attribution_list, f, indent=2, ensure_ascii=False, default=to_serializable)
    if name == 'nt':
        path_to_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    else:
        path_to_wkhtmltopdf = r'/usr/bin/wkhtmltopdf'
    config_wkhtml = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)
    pdfkit.from_string(html_output.data, os.path.join(workdir, f'word_importance_{args.feature_col}.pdf'), configuration=config_wkhtml)
    print("Finish interpreting...")

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze = 'freeze' if args.freeze_embedding else 'unfreeze'
    
    # Create loss and class weight handler string for folder structure
    loss_cw_handler = f"{args.loss_fn}-{args.class_weight}"
    
    workdir = os.path.join(args.output_dir, args.scenario, args.feature_col, args.embedding, freeze, loss_cw_handler, str(args.seed))
    print(f'current scenario: {workdir}')
    os.makedirs(workdir, exist_ok=True)
    if os.path.exists(os.path.join(workdir, 'scenario_arguments.json')) and not args.overwrite:
        print('Scenario has been executed. Skipped!')
        return exit(0)
    train_df, test_df, label_encoder = prepare_data(args)
    # Expose encoder classes on args so downstream reporting can save them
    args.label_classes = label_encoder.classes_.tolist()
    embedding_model, tokenizer = get_model_and_tokenizer(args, device)
    merged_df, merged_loader, train_loader, test_loader = get_dataloaders(train_df, test_df, tokenizer, args)
    model_class = MODEL_REGISTRY[args.scenario]
    model = model_class(embedding_model, tokenizer, freeze_embedding=args.freeze_embedding).to(device)
    # Build class_map from LabelEncoder classes to ensure alignment
    global pro2idx, idx2pro
    pro_classes = label_encoder.classes_.tolist()
    pro2idx = {name: idx for idx, name in enumerate(pro_classes)}
    idx2pro = {idx: name for idx, name in enumerate(pro_classes)}
    class_map = pro2idx

    best_model_state, best_epoch, best_f1_epoch, best_acc_epoch = train(model, train_loader, test_loader, train_df, args, device, class_map=class_map)
    best_model = model_class(embedding_model, tokenizer, freeze_embedding=args.freeze_embedding).to(device)
    best_model.load_state_dict(best_model_state)
    prediction_df, preds_decoded, tests_decoded = evaluate(best_model, test_loader, test_df, args, device)
    report_results(prediction_df, preds_decoded, tests_decoded, workdir, args, best_epoch, best_f1_epoch, best_acc_epoch)
    visualize_and_save(merged_loader, idx2pro, best_model, device, workdir, merged_df, args, best_model_state)
    torch.backends.cudnn.enabled=False
    interpretability_report(best_model, tokenizer, args, device, workdir)
    return exit(0)


if __name__ == "__main__":
    main()