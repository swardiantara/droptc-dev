import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def cluster_sentences(sentences: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a unique integer ID to each unique sentence in the 'sentence' column.

    Args:
        sentences: A Pandas DataFrame containing the sentences to be clustered.
    Returns:
        A Pandas DataFrame with an added 'cluster_id' column.
    """
    # The factorize method provides a simple way to get unique integer IDs for each unique sentence.
    # It returns a tuple of (codes, uniques). We only need the codes.
    sentences['cluster_id'] = pd.factorize(sentences['sentence'])[0]
    return sentences


def summarize_evidence(evidence_df: pd.DataFrame, output_dir: str):
    """
    Summarizes the evidence dataframe by counting the frequency of 'problem_type' and 'cluster_id',
    and exports the summaries as graphs in PDF files.

    Args:
        evidence_df: A Pandas DataFrame containing the evidence data, including 'problem_type' and 'cluster_id'.
        output_dir: The directory where the output PDF files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Summarize 'problem_type' ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a figure and axes for the problem_type plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    problem_type_counts = evidence_df['problem_type'].value_counts()
    sns.barplot(x=problem_type_counts.index, y=problem_type_counts.values, ax=ax1, palette='viridis')
    
    ax1.set_title(f'Frequency of Problem Types', fontsize=16)
    ax1.set_xlabel('Problem Type', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    # Save the problem_type plot
    problem_type_output_path = os.path.join(output_dir, f"problem_type_summary.pdf")
    fig1.savefig(problem_type_output_path)
    plt.close(fig1)
    print(f"Saved problem type summary to {problem_type_output_path}")

    # --- Summarize 'cluster_id' ---
    # Create a figure and axes for the cluster_id plot
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    cluster_id_counts = evidence_df['cluster_id'].value_counts().nlargest(20)  # Top 20 most frequent
    
    # We need to get the sentence for the legend
    # Create a mapping from cluster_id to the first sentence found for that cluster
    cluster_to_sentence = evidence_df.drop_duplicates(subset='cluster_id').set_index('cluster_id')['sentence']
    
    # Get the labels for the y-axis
    y_labels = [f"Cluster {i}" for i in cluster_id_counts.index]

    sns.barplot(x=cluster_id_counts.values, y=y_labels, ax=ax2, palette='plasma', orient='h')

    ax2.set_title(f'Top 20 Most Frequent Log Sentences (by Cluster ID)', fontsize=16)
    ax2.set_xlabel('Frequency', fontsize=12)
    ax2.set_ylabel('Cluster ID', fontsize=12)
    
    # Create a legend with the sentence for each cluster
    legend_elements = [plt.Rectangle((0, 0), 1, 1, color=sns.color_palette('plasma', 20)[i], label=f'Cluster {cluster_id_counts.index[i]}: {cluster_to_sentence[cluster_id_counts.index[i]]}') for i in range(len(cluster_id_counts))]
    ax2.legend(handles=legend_elements, title="Sentences", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for the legend

    # Save the cluster_id plot
    cluster_id_output_path = os.path.join(output_dir, f"cluster_id_summary.pdf")
    fig2.savefig(cluster_id_output_path)
    plt.close(fig2)
    print(f"Saved cluster ID summary to {cluster_id_output_path}")