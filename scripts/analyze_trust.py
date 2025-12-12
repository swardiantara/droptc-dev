import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
# from src.droptc.train_classifier import idx2pro, pro2idx

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

class TrustworthinessAnalyzer:
    def __init__(self, base_path='experiments', output_path='visualization/trustworthiness'):
        self.base_path = Path(base_path)
        self.results = {}
        self.output_path = output_path

        if os.path.exists(self.output_path) is False:
            os.makedirs(self.output_path, exist_ok=True)
        
    def load_predictions(self, method, embedding, freeze, loss_fc, class_weight, seeds=range(10)):
        """Load predictions from all seeds for a given method/embedding/freeze combination"""
        predictions = []
        for seed in seeds:
            file_path = self.base_path / method / 'sentence' / embedding / freeze / f'{loss_fc}-{class_weight}' / str(seed) / 'prediction.xlsx'
            if file_path.exists():
                df = pd.read_excel(file_path)
                df['seed'] = seed
                df['method'] = method
                df['embedding'] = embedding
                df['freeze'] = freeze
                df['loss_fc'] = loss_fc
                df['class_weight'] = class_weight
                predictions.append(df)
        
        if predictions:
            return pd.concat(predictions, ignore_index=True)
        return None
    
    def load_all_methods(self, methods_config):
        """
        Load all methods based on configuration
        methods_config example:
        {
            'droptc': [
                ('bert-base-uncased', 'frozen'),
                ('all-MiniLM-L6-v2', 'frozen'),
                ('all-MiniLM-L6-v2', 'unfrozen'),
                ...
            ],
            'baseline1': [('embedding_name', 'frozen')],
            'baseline2': [('embedding_name', 'frozen')],
            ...
        }
        """
        seeds=[14298463, 24677315, 37622020, 43782163, 52680723, 67351593, 70681460, 87212562, 90995999, 99511865]
        all_data = []
        for method, configs in methods_config.items():
            for embedding, freeze, loss_fc, class_weight in configs:
                df = self.load_predictions(method, embedding, freeze, loss_fc, class_weight, seeds)
                if df is not None:
                    all_data.append(df)
        
        if all_data:
            self.results = pd.concat(all_data, ignore_index=True)
            return self.results
        return None
    
    def calculate_ece(self, labels, probs, preds, n_bins=10):
        """
        Calculate Expected Calibration Error for multiclass
        Uses the predicted class probability as confidence
        """
        # Binary correctness: 1 if prediction matches label, 0 otherwise
        correct = (preds == labels).astype(int)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def get_calibration_data(self, labels, probs, preds, n_bins=10):
        """Get calibration data for reliability diagram (multiclass)"""
        correct = (preds == labels).astype(int)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences = []
        accuracies = []
        counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.sum()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = probs[in_bin].mean()
                confidences.append(avg_confidence_in_bin)
                accuracies.append(accuracy_in_bin)
                counts.append(prop_in_bin)
        
        return np.array(confidences), np.array(accuracies), np.array(counts)
    
    def compute_metrics_per_method(self):
        """Compute metrics for each method across all runs"""
        metrics = []
        
        grouped = self.results.groupby(['method', 'embedding', 'freeze', 'loss_fc', 'class_weight', 'seed'])
        
        for (method, embedding, freeze, loss_fc, class_weight, seed), group in grouped:
            labels = group['label'].values
            preds = group['pred'].values
            probs = group['prob'].values
            
            # Calculate metrics
            f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
            precision_weighted = precision_score(labels, preds, average='weighted', zero_division=0)
            recall_weighted = recall_score(labels, preds, average='weighted', zero_division=0)
            acc = accuracy_score(labels, preds)
            ece = self.calculate_ece(labels, probs, preds)
            
            metrics.append({
                'method': method,
                'embedding': embedding,
                'freeze': freeze,
                'loss_fc': loss_fc,
                'class_weight': class_weight,
                'seed': seed,
                'f1_weighted': f1_weighted,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'accuracy': acc,
                'ece': ece
            })
        
        return pd.DataFrame(metrics)
    
    def plot_training_strategy_comparison(self, fine_tuned_embeddings=['all-MiniLM-L6-v2', 'all-mpnet-base-v2'], 
                                         method='droptc'):
        """Compare frozen vs unfrozen for fine-tuned embeddings"""
        metrics_df = self.compute_metrics_per_method()
        
        # Filter for the proposed method and fine-tuned embeddings only
        ft_data = metrics_df[
            (metrics_df['method'] == method) & 
            (metrics_df['embedding'].isin(fine_tuned_embeddings))
        ]
        
        if ft_data.empty:
            print(f"No data found for method '{method}' with fine-tuned embeddings")
            return
        
        # Aggregate across seeds
        agg_data = ft_data.groupby(['embedding', 'freeze']).agg({
            'f1_weighted': ['mean', 'std'],
            'ece': ['mean', 'std']
        }).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot F1 score
        for emb in fine_tuned_embeddings:
            emb_data = agg_data[agg_data['embedding'] == emb]
            if emb_data.empty:
                continue
            x_pos = [0, 1] if emb == fine_tuned_embeddings[0] else [0.25, 1.25]
            
            axes[0].bar(x_pos, 
                       emb_data['f1_weighted']['mean'].values,
                       yerr=emb_data['f1_weighted']['std'].values,
                       width=0.2,
                       label=emb,
                       capsize=5)
        
        axes[0].set_ylabel('Weighted F1 Score')
        axes[0].set_title('Training Strategy Impact on F1 Score')
        axes[0].set_xticks([0.125, 1.125])
        axes[0].set_xticklabels(['Frozen Embedding', 'Fine-tuned Embedding'])
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot ECE
        for emb in fine_tuned_embeddings:
            emb_data = agg_data[agg_data['embedding'] == emb]
            if emb_data.empty:
                continue
            x_pos = [0, 1] if emb == fine_tuned_embeddings[0] else [0.25, 1.25]
            
            axes[1].bar(x_pos, 
                       emb_data['ece']['mean'].values,
                       yerr=emb_data['ece']['std'].values,
                       width=0.2,
                       label=emb,
                       capsize=5)
        
        axes[1].set_ylabel('Expected Calibration Error (ECE)')
        axes[1].set_title('Training Strategy Impact on Calibration')
        axes[1].set_xticks([0.125, 1.125])
        axes[1].set_xticklabels(['Frozen Embedding', 'Fine-tuned Embedding'])
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'training_strategy_comparison.pdf'), dpi=300, bbox_inches='tight')
        # plt.show()
        
        return agg_data
    
    def plot_ece_comparison(self, top_n=None, sort_by='ece'):
        """Plot ECE comparison across all methods"""
        metrics_df = self.compute_metrics_per_method()
        
        # Aggregate across seeds
        agg_data = metrics_df.groupby(['method', 'embedding', 'freeze', 'loss_fc', 'class_weight']).agg({
            'ece': ['mean', 'std'],
            'f1_weighted': ['mean', 'std']
        }).reset_index()
        
        agg_data['label'] = agg_data.apply(
            lambda x: f"{x['method']}\n{x['embedding'][:20]}\n({x['freeze']})", axis=1
        )
        
        # Sort by mean ECE or F1
        sort_col = (sort_by, 'mean')
        ascending = True if sort_by == 'ece' else False
        agg_data = agg_data.sort_values(sort_col, ascending=ascending)
        
        if top_n:
            agg_data = agg_data.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 3.5))
        
        x_pos = np.arange(len(agg_data))
        colors = ['#1f77b4' if 'droptc' in label else '#ff7f0e' for label in agg_data['label']]
        
        ax.bar(x_pos, 
               agg_data[('ece', 'mean')].values,
               yerr=agg_data[('ece', 'std')].values,
               capsize=5,
               alpha=0.7,
               color=colors)
        
        ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=12)
        ax.set_title('Calibration Comparison Across Methods (Lower is Better)', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agg_data['label'].values, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', label='DROPTC (Proposed)'),
                          Patch(facecolor='#ff7f0e', label='Baselines')]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'ece_comparison.pdf'), dpi=300, bbox_inches='tight')
        # plt.show()
        
        return agg_data
    
    def plot_reliability_diagrams(self, methods_to_plot):
        """
        Plot reliability diagrams for specified methods
        methods_to_plot: list of tuples (method, embedding, freeze, label)
        """
        n_methods = len(methods_to_plot)
        # Preferred layout: 2 rows x 3 columns (up to 6 plots)
        # If more than 6 methods are requested, add more rows as needed.
        n_cols = 3
        n_rows = 2 if n_methods <= 6 else int(np.ceil(n_methods / n_cols))
        # Share x and y axes across subplots for direct comparison
        # Use a compact figure size that fits a single manuscript column (approx 3.3 inches wide)
        # col_width_in = 3.3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 1.5 * n_rows), sharex=True, sharey=True)
        # Flatten axes array for easy indexing
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
        
        for idx, (method, embedding, freeze, loss_fc, class_weight, label) in enumerate(methods_to_plot):
            # Filter data for this method
            method_data = self.results[
                (self.results['method'] == method) &
                (self.results['embedding'] == embedding) &
                (self.results['freeze'] == freeze) &
                (self.results['loss_fc'] == loss_fc) &
                (self.results['class_weight'] == class_weight)
            ]
            
            if method_data.empty:
                print(f"No data for {label}")
                continue
            
            # Aggregate across all seeds
            labels = method_data['label'].values
            preds = method_data['pred'].values
            probs = method_data['prob'].values
            
            confidences, accuracies, counts = self.get_calibration_data(labels, probs, preds)
            ece = self.calculate_ece(labels, probs, preds)

            # Plot into the flattened axes
            ax = axes_flat[idx]
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.0)
            # Use light gray for bars for clarity
            ax.bar(confidences, accuracies, width=0.08, alpha=0.9,
                   color='lightgray', edgecolor='black')

            # Only show x-label on bottom row
            if idx // n_cols == n_rows - 1:
                ax.set_xlabel('Confidence', fontsize=9)
                # ax.set_xticklabels()
            else:
                ax.set_xlabel('')
                # ax.set_xticklabels([])

            ax.set_ylabel('Accuracy', fontsize=9)
            ax.set_title(f'{label}\nECE: {ece:.4f}', fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            
        # Turn off any unused subplots
        if len(axes_flat) > n_methods:
            for ax in axes_flat[n_methods:]:
                ax.axis('off')

        # Create a single shared legend (place below the plots)
        legend_elements = [Line2D([0], [0], color='k', lw=1, linestyle='--', label='Perfect calibration'),
                           Patch(facecolor='lightgray', edgecolor='black', label='Model')]
        # Place legend in the lower center just below the subplots
        fig.legend(handles=legend_elements, loc='lower center', ncols=2, fontsize=8, bbox_to_anchor=(0.5, -0.01))

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.18)
        plt.savefig(os.path.join(self.output_path, 'reliability_diagrams_small.pdf'), dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_confidence_stratified_performance(self, methods_to_plot, n_bins=5):
        """Plot F1 score at different confidence levels"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method, embedding, freeze, label in methods_to_plot:
            method_data = self.results[
                (self.results['method'] == method) &
                (self.results['embedding'] == embedding) &
                (self.results['freeze'] == freeze)
            ]
            
            if method_data.empty:
                continue
            
            labels = method_data['label'].values
            preds = method_data['pred'].values
            probs = method_data['prob'].values
            
            # Create bins
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            f1_scores = []
            for i in range(n_bins):
                mask = (probs >= bin_edges[i]) & (probs < bin_edges[i+1])
                if mask.sum() > 0:
                    f1 = f1_score(labels[mask], preds[mask], average='weighted', zero_division=0)
                    f1_scores.append(f1)
                else:
                    f1_scores.append(np.nan)
            
            ax.plot(bin_centers, f1_scores, marker='o', label=label, linewidth=2, markersize=8)
        
        ax.set_xlabel('Confidence Bin', fontsize=12)
        ax.set_ylabel('Weighted F1 Score', fontsize=12)
        ax.set_title('Performance vs. Confidence Level', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'confidence_stratified_performance.pdf'), dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_coverage_f1_curve(self, methods_to_plot):
        """Plot coverage vs F1 curves (selective prediction)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method, embedding, freeze, label in methods_to_plot:
            method_data = self.results[
                (self.results['method'] == method) &
                (self.results['embedding'] == embedding) &
                (self.results['freeze'] == freeze)
            ]
            
            if method_data.empty:
                continue
            
            labels = method_data['label'].values
            preds = method_data['pred'].values
            probs = method_data['prob'].values
            
            # Sort by confidence (descending)
            sorted_indices = np.argsort(probs)[::-1]
            sorted_labels = labels[sorted_indices]
            sorted_preds = preds[sorted_indices]
            
            # Calculate cumulative F1 score
            coverages = []
            f1_scores = []
            
            # Sample points for efficiency (every 5% of data)
            step = max(1, len(sorted_labels) // 20)
            
            for i in range(step, len(sorted_labels) + 1, step):
                coverage = i / len(sorted_labels)
                f1 = f1_score(sorted_labels[:i], sorted_preds[:i], average='weighted', zero_division=0)
                coverages.append(coverage)
                f1_scores.append(f1)
            
            # Ensure we include the last point
            if coverages[-1] < 1.0:
                coverages.append(1.0)
                f1_scores.append(f1_score(sorted_labels, sorted_preds, average='weighted', zero_division=0))
            
            ax.plot(coverages, f1_scores, label=label, linewidth=2)
        
        ax.set_xlabel('Coverage (fraction of predictions retained)', fontsize=12)
        ax.set_ylabel('Weighted F1 Score', fontsize=12)
        ax.set_title('Coverage vs. F1 Score (Selective Prediction)', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'coverage_f1_curve.pdf'), dpi=300, bbox_inches='tight')
        # plt.show()
    
    def plot_cross_run_stability(self, methods_to_plot):
        """Analyze prediction variance across runs"""
        results_list = []
        
        for method, embedding, freeze, loss_fc, class_weight, label in methods_to_plot:
            method_data = self.results[
                (self.results['method'] == method) &
                (self.results['embedding'] == embedding) &
                (self.results['freeze'] == freeze) &
                (self.results['loss_fc'] == loss_fc) &
                (self.results['class_weight'] == class_weight)
            ]
            
            if method_data.empty:
                continue
            
            # Calculate variance across runs for each sample
            sample_col = 'sentence' if 'sentence' in method_data.columns else 'message'
            
            # Group by sample and calculate statistics
            variance_data = method_data.groupby(sample_col).agg({
                'prob': ['mean', 'std', 'count'],
                'pred': lambda x: x.nunique()  # Number of different predictions
            })
            variance_data.columns = ['prob_mean', 'prob_std', 'count', 'pred_diversity']
            
            # Only samples present in all runs
            variance_data = variance_data[variance_data['count'] == 10]
            
            results_list.append({
                'label': label,
                'mean_prob_std': variance_data['prob_std'].mean(),
                'std_prob_std': variance_data['prob_std'].std(),
                'high_variance_samples': (variance_data['prob_std'] > 0.1).sum(),
                'total_samples': len(variance_data),
                'mean_pred_diversity': variance_data['pred_diversity'].mean()
            })
        
        results_df = pd.DataFrame(results_list)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.5))
        
        # Mean variance in probability
        axes[0].bar(range(len(results_df)), results_df['mean_prob_std'],
                    yerr=results_df['std_prob_std'], capsize=5, alpha=0.7, color='lightgray')
        axes[0].set_ylabel('Mean Std Dev', fontsize=10)
        axes[0].set_title('Cross-Run Prediction Stability\n(Confidence Scores)', fontsize=10)
        axes[0].set_xticks(range(len(results_df)))
        axes[0].set_xticklabels(results_df['label'], rotation=30, ha='right')
        axes[0].grid(axis='y', alpha=0.3)

        # High variance samples
        axes[1].bar(range(len(results_df)),
                    results_df['high_variance_samples'] / results_df['total_samples'] * 100,
                    alpha=0.7, color='lightgray')
        axes[1].set_ylabel('Number of Samples\nwith High Variance', fontsize=10)
        axes[1].set_title('Unstable Prediction Samples\n(std > 0.1)', fontsize=10)
        axes[1].set_xticks(range(len(results_df)))
        axes[1].set_xticklabels(results_df['label'], rotation=30, ha='right')
        axes[1].grid(axis='y', alpha=0.3)

        # Prediction diversity
        axes[2].bar(range(len(results_df)), results_df['mean_pred_diversity'], alpha=0.7, color='lightgray')
        axes[2].set_ylabel('Avg. Number of Preds\nAcross 10 Runs', fontsize=10)
        axes[2].set_title('Prediction Diversity Across Runs', fontsize=10)
        axes[2].set_xticks(range(len(results_df)))
        axes[2].set_xticklabels(results_df['label'], rotation=30, ha='right')
        axes[2].grid(axis='y', alpha=0.3)
        axes[2].axhline(y=1, color='green', linestyle='--', label='Perfect consistency', alpha=0.7)
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'cross_run_stability_small.pdf'), dpi=300, bbox_inches='tight')
        # plt.show()
        
        return results_df
    
    def plot_confidence_distributions(self, methods_to_plot):
        """Plot confidence distributions for correct vs incorrect predictions"""
        n_methods = len(methods_to_plot)

        # Use a 2x3 grid for up to 6 methods (2 rows, 3 columns). If more methods are
        # provided, add rows dynamically.
        n_cols = 3
        n_rows = 2 if n_methods <= 6 else int(np.ceil(n_methods / n_cols))

        # Share axes across subplots for direct comparisons
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 1.5 * n_rows), sharex=True, sharey=True)
        # Flatten axes for easy indexing
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        for idx, (method, embedding, freeze, loss_fc, class_weight, label) in enumerate(methods_to_plot):
            ax = axes_flat[idx]
            method_data = self.results[
                (self.results['method'] == method) &
                (self.results['embedding'] == embedding) &
                (self.results['freeze'] == freeze) &
                (self.results['loss_fc'] == loss_fc) &
                (self.results['class_weight'] == class_weight)
            ]

            if method_data.empty:
                print('confidence distribution: No data for', label)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(label)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            # print(f'Plotting confidence distribution for {label}, total samples: {len(method_data)}')
            # print(method_data['verdict'].value_counts())
            # Separate correct and incorrect predictions
            correct = method_data[method_data['verdict'] == True]
            incorrect = method_data[method_data['verdict'] == False]
            if correct.empty or incorrect.empty:
                # print(f"Correct: {correct.empty}, Incorrect: {incorrect.empty}")
                print('confidence distribution: Not enough data for', label)
                # ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
                # ax.set_title(label)
            # KDE plots for correct vs incorrect confidences
            correct_probs = correct['prob'].dropna().values
            incorrect_probs = incorrect['prob'].dropna().values

            # Use smaller bandwidth to resolve peaks near 1.0
            bw = 0.3

            # Plot KDEs (or fall back to histogram/rug if not enough points)
            if len(correct_probs) > 1:
                try:
                    sns.kdeplot(correct_probs, ax=ax, fill=True, bw_adjust=bw,
                                label='Correct', color='green', alpha=0.45)
                except Exception:
                    ax.hist(correct_probs, bins=20, density=True, alpha=0.45, color='green', label='Correct')
            elif len(correct_probs) == 1:
                ax.plot(correct_probs, [0], '|', color='green', markersize=12, label='Correct')

            if len(incorrect_probs) > 1:
                try:
                    sns.kdeplot(incorrect_probs, ax=ax, fill=True, bw_adjust=bw,
                                label='Incorrect', color='red', alpha=0.45)
                except Exception:
                    ax.hist(incorrect_probs, bins=20, density=True, alpha=0.45, color='red', label='Incorrect')
            elif len(incorrect_probs) == 1:
                ax.plot(incorrect_probs, [0], '|', color='red', markersize=12, label='Incorrect')

            ax.set_xlim(0, 1)
            ax.set_xlabel('Confidence', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_title(f'{label}', fontsize=10)
            # Ensure legend is only shown on the first subplot (lower-left)
            legend_elements = [Patch(facecolor='green', edgecolor='black', label='Correct'),
                               Patch(facecolor='red', edgecolor='black', label='Incorrect')]
            if idx == 0:
                ax.legend(handles=legend_elements, fontsize=8, loc='lower left', ncol=2)
            ax.grid(alpha=0.3)

            # Rug plot to show raw points density
            if len(correct_probs) > 0:
                ax.plot(correct_probs, np.full_like(correct_probs, -0.005), '|', color='green', alpha=0.5)
            if len(incorrect_probs) > 0:
                ax.plot(incorrect_probs, np.full_like(incorrect_probs, -0.01), '|', color='red', alpha=0.5)

            # Inset zoom over 0.8-1.0 to show dense region
            try:
                axins = inset_axes(ax, width="40%", height="30%", loc='upper right')
                if len(correct_probs) > 1:
                    sns.kdeplot(correct_probs, ax=axins, fill=True, bw_adjust=bw, color='green', alpha=0.45, legend=False)
                if len(incorrect_probs) > 1:
                    sns.kdeplot(incorrect_probs, ax=axins, fill=True, bw_adjust=bw, color='red', alpha=0.45, legend=False)
                axins.set_xlim(0.9, 1.1)
                axins.set_ylim(0, None)
                axins.set_xticks([0.8, 0.9, 1.0])
                axins.set_yticks([])
            except Exception:
                pass

            # Annotate mean and std for each distribution
            try:
                if len(correct_probs) > 0:
                    mu_c = np.mean(correct_probs)
                    sd_c = np.std(correct_probs)
                    ax.text(0.02, 0.95, f'μ={mu_c:.3f}\nσ={sd_c:.3f}', transform=ax.transAxes,
                            ha='left', va='top', fontsize=8, color='green', bbox=dict(facecolor='white', alpha=0.7))
                if len(incorrect_probs) > 0:
                    mu_i = np.mean(incorrect_probs)
                    sd_i = np.std(incorrect_probs)
                    # Place incorrect annotation below the correct one on the left to avoid overlap
                    ax.text(0.02, 0.60, f'μ={mu_i:.3f}\nσ={sd_i:.3f}', transform=ax.transAxes,
                            ha='left', va='top', fontsize=8, color='red', bbox=dict(facecolor='white', alpha=0.7))
            except Exception:
                pass

        # Turn off unused subplots
        total_subplots = n_rows * n_cols
        if total_subplots > n_methods:
            for ax in axes_flat[n_methods:total_subplots]:
                ax.axis('off')

        plt.tight_layout()
        # Save high-resolution PDF for paper
        plt.savefig(os.path.join(self.output_path, 'confidence_distributions_small.pdf'), dpi=600, bbox_inches='tight')
        # plt.show()
    
    def plot_per_class_calibration(self, methods_to_plot, n_classes=7):
        """Analyze calibration per class (useful for imbalanced datasets)"""
        for method, embedding, freeze, loss_fc, class_weight, label in methods_to_plot:
            method_data = self.results[
                (self.results['method'] == method) &
                (self.results['embedding'] == embedding) &
                (self.results['freeze'] == freeze) &
                (self.results['loss_fc'] == loss_fc) &
                (self.results['class_weight'] == class_weight)
            ]
            
            if method_data.empty:
                continue
            
            fig, axes = plt.subplots(1, n_classes, figsize=(4*n_classes, 3))
            
            for class_idx in [key for key, _ in idx2pro.items()]:
                # Filter predictions for this class
                class_data = method_data[method_data['pred'] == idx2pro[class_idx]]
                
                if len(class_data) == 0:
                    axes[class_idx].text(0.5, 0.5, 'No predictions', 
                                        ha='center', va='center', transform=axes[class_idx].transAxes)
                    axes[class_idx].set_title(f'Class {class_idx}')
                    continue
                
                labels = (class_data['label'] == class_idx).astype(int).values
                probs = class_data['prob'].values
                
                confidences, accuracies, counts = self.get_calibration_data(
                    labels, probs, np.ones_like(labels), n_bins=5
                )
                
                axes[class_idx].plot([0, 1], [0, 1], 'k--', linewidth=1.5)
                axes[class_idx].bar(confidences, accuracies, width=0.15, alpha=0.7, edgecolor='black')
                axes[class_idx].set_title(f'Class {idx2pro[class_idx]}\n(n={len(class_data)})', fontsize=10)
                axes[class_idx].set_xlabel('Confidence', fontsize=9)
                if class_idx == 0:
                    axes[class_idx].set_ylabel('Precision', fontsize=9)
                axes[class_idx].set_xlim([0, 1])
                axes[class_idx].set_ylim([0, 1])
                axes[class_idx].grid(alpha=0.3)
            
            plt.suptitle(f'{label} - Per-Class Calibration', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, f'per_class_calibration_{label.replace(" ", "_")}.pdf'), dpi=300, bbox_inches='tight')
            # plt.show()
    
    def generate_summary_table(self):
        """Generate summary table with all key metrics"""
        metrics_df = self.compute_metrics_per_method()
        
        summary = metrics_df.groupby(['method', 'embedding', 'freeze', 'loss_fc', 'class_weight']).agg({
            'accuracy': ['mean', 'std'],
            'f1_weighted': ['mean', 'std'],
            'precision_weighted': ['mean', 'std'],
            'recall_weighted': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'ece': ['mean', 'std']
        }).reset_index()
        
        summary.columns = ['method', 'embedding', 'freeze', 'loss_fc', 'class_weight',
                          'f1_weighted_mean', 'f1_weighted_std',
                          'precision_weighted_mean', 'precision_weighted_std',
                          'recall_weighted_mean', 'recall_weighted_std',
                          'accuracy_mean', 'accuracy_std',
                          'ece_mean', 'ece_std']
        
        # Format for display
        # summary['f1_weighted'] = summary.apply(
        #     lambda x: f"{x['f1_weighted_mean']:.4f} ± {x['f1_weighted_std']:.4f}", axis=1
        # )
        # summary['precision_weighted'] = summary.apply(
        #     lambda x: f"{x['precision_weighted_mean']:.4f} ± {x['precision_weighted_std']:.4f}", axis=1
        # )
        # summary['recall_weighted'] = summary.apply(
        #     lambda x: f"{x['recall_weighted_mean']:.4f} ± {x['recall_weighted_std']:.4f}", axis=1
        # )
        # summary['accuracy'] = summary.apply(
        #     lambda x: f"{x['accuracy_mean']:.4f} ± {x['accuracy_std']:.4f}", axis=1
        # )
        # summary['ece'] = summary.apply(
        #     lambda x: f"{x['ece_mean']:.4f} ± {x['ece_std']:.4f}", axis=1
        # )
        
        # display_summary = summary[['method', 'embedding', 'freeze', 'loss_fc', 'class_weight',
        #                            'f1_weighted', 'precision_weighted', 
        #                            'recall_weighted', 'accuracy', 'ece']]
        
        # Sort by F1 score
        display_summary = summary.sort_values('f1_mean', ascending=False) if 'f1_mean' in summary.columns else summary
        
        # Save to CSV
        display_summary.to_excel(os.path.join(self.output_path, 'trustworthiness_summary.xlsx'), index=False)
        print(f"Summary table saved to {os.path.join(self.output_path, 'trustworthiness_summary.xlsx')}")
        
        return display_summary
    
    def compute_statistical_comparison(self, comparisons):
        """
        Compute statistical comparisons for Table 8.
        
        comparisons: list of tuples (name, scenario1, scenario2)
        Example:
        """
        results = []
        comp_dict = dict()
        for idx, (method, embedding, freeze, loss_fc, class_weight, label) in enumerate(comparisons):
            # Filter data for this method
            method_data = self.results[
                (self.results['method'] == method) &
                (self.results['embedding'] == embedding) &
                (self.results['freeze'] == freeze) &
                (self.results['loss_fc'] == loss_fc) &
                (self.results['class_weight'] == class_weight)
            ]
            if method_data.empty:
                print(f"No data for {label}")
                continue
            print(f"Preparing data for comparison: {label}, total samples: {(method_data.columns)}")
            comp_dict[label] = method_data
            print(comp_dict.keys())

        scenarios = [
            ('Fine-tuned vs Pre-trained (Frozen)', 'MPNet-FT-Frozen', 'MPNet-PT-Frozen'),
            ('Fine-tuned vs Pre-trained (FullFT)', 'MPNet-FT-FullFT', 'MPNet-PT-FullFT'),
            ('Full FT vs Frozen (Pre-trained)', 'MPNet-PT-FullFT', 'MPNet-PT-Frozen'),
            ('Full FT vs Frozen (Fine-tuned)', 'MPNet-FT-FullFT', 'MPNet-FT-Frozen'),
            ('DroPTC-WoCW vs DroPTC', 'MPNet-FT-FullFT', 'DroPTC'),
            ('DroPTC vs DroPTC-WoCW', 'DroPTC', 'MPNet-FT-FullFT'),
        ]
        for comp_name, scenario1, scenario2 in scenarios:
            df1 = comp_dict[scenario1].sort_values(by=['seed']).reset_index(drop=True)
            df2 = comp_dict[scenario2].sort_values(by=['seed']).reset_index(drop=True)
            
            # Get mean confidence per seed
            conf1_per_seed = df1.groupby('seed')['prob'].mean()
            conf2_per_seed = df2.groupby('seed')['prob'].mean()
            
            # Paired t-test (since same test set across seeds)
            t_stat, p_value = stats.ttest_rel(conf1_per_seed, conf2_per_seed)
            
            # Effect size (Cohen's d for paired samples)
            diff = conf1_per_seed - conf2_per_seed
            cohens_d = diff.mean() / diff.std()
            
            # Delta confidence
            delta_conf = conf1_per_seed.mean() - conf2_per_seed.mean()
            
            results.append({
                'Comparison': comp_name,
                'Delta Confidence': f"{delta_conf:+.4f}",
                "Cohen's d": f"{cohens_d:.3f}",
                'p-value': f"<0.001" if p_value < 0.001 else f"{p_value:.4f}"
            })
        
        results_df = pd.DataFrame(results)
        print("\nStatistical Comparison Table:")
        print(results_df.to_string(index=False))
        results_df.to_excel(os.path.join(self.output_path, 'confidence_statistical_test.xlsx'), index=False)
        # return results_df


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TrustworthinessAnalyzer(base_path='experiments')
    
    # Define your methods configuration
    # Adjust this based on your actual directory structure
    # [
    #     ('Fine-tuned vs Pre-trained (Frozen)', 'MPNet-FT-Frozen', 'MPNet-PT-Frozen'),
    #     ('Fine-tuned vs Pre-trained (FullFT)', 'MPNet-FT-FullFT', 'MPNet-PT-FullFT'),
    #     ('Full FT vs Frozen (Pre-trained)', 'MPNet-PT-FullFT', 'MPNet-PT-Frozen'),
    #     ('Full FT vs Frozen (Fine-tuned)', 'MPNet-FT-FullFT', 'MPNet-FT-Frozen'),
    # ]
    methods_config = {
        # 'MPNet-PT-Frozen': [('all-mpnet-base-v2', 'freeze', 'ce', 'uniform')],
        # 'MPNet-FT-Frozen': [('DroPTC-all-mpnet-base-v2-sentence', 'freeze', 'ce', 'uniformold')],
        # 'MPNet-PT-FullFT': [('all-mpnet-base-v2', 'unfreeze', 'ce', 'uniform')],
        # 'MPNet-FT-FullFT': [('DroPTC-all-mpnet-base-v2-sentence', 'unfreeze', 'ce', 'uniformold')],
        # 'MPNet-Frozen': {'method': 'droptc', 'embedding': 'all-mpnet-base-v2', 'freeze': 'freeze'},
        # 'MPNet-FullFT': {'method': 'droptc', 'embedding': 'all-mpnet-base-v2', 'freeze': 'unfreeze'},
        # 'MPNet-FT-Frozen': {'method': 'droptc', 'embedding': 'DroPTC-all-mpnet-base-v2-sentence', 'freeze': 'freeze'},
        # 'MPNet-FT-FullFT': {'method': 'droptc', 'embedding': 'DroPTC-all-mpnet-base-v2-sentence', 'freeze': 'unfreeze'},
        'droptc': [
            ('bert-base-uncased', 'freeze', 'ce', 'uniform'),
            ('bert-base-uncased', 'unfreeze', 'ce', 'uniform'),
            ('neo-bert', 'freeze', 'ce', 'uniform'),
            ('neo-bert', 'unfreeze', 'ce', 'uniform'),
            ('modern-bert', 'freeze', 'ce', 'uniform'),
            ('modern-bert', 'unfreeze', 'ce', 'uniform'),
            ('all-MiniLM-L6-v2', 'freeze', 'ce', 'uniform'),
            ('all-MiniLM-L6-v2', 'unfreeze', 'ce', 'uniform'),  
            ('all-mpnet-base-v2', 'freeze', 'ce', 'uniform'),
            ('all-mpnet-base-v2', 'unfreeze', 'ce', 'uniform'),  
            ('DroPTC-all-MiniLM-L6-v2-sentence', 'freeze', 'focal', 'inverse'), # fine-tuned
            ('DroPTC-all-MiniLM-L6-v2-sentence', 'unfreeze', 'ce', 'uniformold'), # fine-tuned
            ('DroPTC-all-mpnet-base-v2-sentence', 'freeze', 'focal', 'inverse'),  # fine-tuned
            ('DroPTC-all-mpnet-base-v2-sentence', 'freeze', 'ce', 'uniformold'),  # fine-tuned
            ('DroPTC-all-mpnet-base-v2-sentence', 'unfreeze', 'ce', 'inverse'),  # fine-tuned
            ('DroPTC-all-mpnet-base-v2-sentence', 'unfreeze', 'ce', 'uniformold'),  # fine-tuned
        ],
        'drolove': [('bert-base-uncased', 'unfreeze', 'ce', 'uniform')],
        'dronelog': [('DroPTC-all-mpnet-base-v2-sentence', 'unfreeze', 'ce', 'uniform')],
        'neurallog': [('bert-base-uncased', 'unfreeze', 'ce', 'uniform')],
        'transsentlog': [('bert-base-uncased', 'unfreeze', 'ce', 'uniform')],
    }
    
    # Load all data
    print("Loading data...")
    analyzer.load_all_methods(methods_config)
    
    # 1. Training Strategy Comparison (addresses your missing analysis)
    print("\n1. Generating training strategy comparison...")
    analyzer.plot_training_strategy_comparison(
        fine_tuned_embeddings=['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'DroPTC-all-MiniLM-L6-v2-sentence', 'DroPTC-all-mpnet-base-v2-sentence'],
        method='droptc'
    )
    
    # 2. ECE Comparison across all methods
    print("\n2. Generating ECE comparison...")
    analyzer.plot_ece_comparison()
    
    # 3. Reliability Diagrams for key models
    print("\n3. Generating reliability diagrams...")
    key_methods = [
        ('droptc', 'DroPTC-all-mpnet-base-v2-sentence', 'unfreeze', 'ce', 'inverse', 'DroPTC'),
        ('droptc', 'DroPTC-all-mpnet-base-v2-sentence', 'unfreeze', 'ce', 'uniformold', 'DroPTC-WoCW'),
        ('dronelog', 'DroPTC-all-mpnet-base-v2-sentence', 'unfreeze', 'ce', 'uniform', 'DroneLog'),
        ('drolove', 'bert-base-uncased', 'unfreeze', 'ce', 'uniform', 'DroLoVe'),
        ('neurallog', 'bert-base-uncased', 'unfreeze', 'ce', 'uniform', 'NeuralLog'),
        ('transsentlog', 'bert-base-uncased', 'unfreeze', 'ce', 'uniform', 'TransSentLog'),
    ]
    analyzer.plot_reliability_diagrams(key_methods)
    
    # 4. Confidence-stratified performance
    print("\n4. Generating confidence-stratified performance...")
    # analyzer.plot_confidence_stratified_performance(key_methods)
    
    # 5. Coverage-F1 curves
    print("\n5. Generating coverage-F1 curves...")
    # analyzer.plot_coverage_f1_curve(key_methods)
    
    # 6. Cross-run stability
    print("\n6. Generating cross-run stability analysis...")
    stability_results = analyzer.plot_cross_run_stability(key_methods)
    print("\nStability Results:")
    print(stability_results)
    
    # 7. Confidence distributions
    print("\n7. Generating confidence distributions...")
    analyzer.plot_confidence_distributions(key_methods)
    
    # 8. Per-class calibration (important for imbalanced data)
    print("\n8. Generating per-class calibration analysis...")
    # Only for the best model to save space
    analyzer.plot_per_class_calibration([
        ('droptc', 'DroPTC-all-mpnet-base-v2-sentence', 'unfreeze', 'ce', 'inverse', 'DroPTC')
    ])

    # 2. Statistical comparison for Q1 and Q2
    comparisons = [
            ('droptc', 'all-mpnet-base-v2', 'freeze', 'ce', 'uniform', 'MPNet-PT-Frozen'),
            ('droptc', 'DroPTC-all-mpnet-base-v2-sentence', 'freeze', 'ce', 'uniformold', 'MPNet-FT-Frozen'),
            ('droptc', 'all-mpnet-base-v2', 'unfreeze', 'ce', 'uniform', 'MPNet-PT-FullFT'),
            ('droptc', 'DroPTC-all-mpnet-base-v2-sentence', 'unfreeze', 'ce', 'uniformold', 'MPNet-FT-FullFT'),
            ('droptc', 'DroPTC-all-mpnet-base-v2-sentence', 'unfreeze', 'ce', 'inverse', 'DroPTC'),
        ]
    # 'MPNet-PT-Frozen': [('all-mpnet-base-v2', 'freeze', 'ce', 'uniform')],
    # 'MPNet-FT-Frozen': [('DroPTC-all-mpnet-base-v2-sentence', 'freeze', 'ce', 'uniformold')],
    # 'MPNet-PT-FullFT': [('all-mpnet-base-v2', 'unfreeze', 'ce', 'uniform')],
    # 'MPNet-FT-FullFT': [('DroPTC-all-mpnet-base-v2-sentence', 'unfreeze', 'ce', 'uniformold')],
    stats_df = analyzer.compute_statistical_comparison(comparisons)
    
    # 9. Generate summary table
    print("\n9. Generating summary table...")
    summary = analyzer.generate_summary_table()
    print("\nSummary Table:")
    print(summary)
    
    print("\nAll visualizations generated successfully!")
    print("\nGenerated files:")
    print("  - training_strategy_comparison.png")
    print("  - ece_comparison.png")
    print("  - reliability_diagrams.png")
    print("  - confidence_stratified_performance.png")
    print("  - coverage_f1_curve.png")
    print("  - cross_run_stability.png")
    print("  - confidence_distributions.png")
    print("  - per_class_calibration_*.png")
    print("  - trustworthiness_summary.csv")