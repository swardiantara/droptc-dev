import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px


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


def create_timeline_chart(evidence_df: pd.DataFrame, output_dir: str):
    """
    Generates a Gantt-style timeline chart to visualize the event log.

    Args:
        evidence_df: A Pandas DataFrame with 'date', 'time', 'problem_type', and 'sentence' columns.
        output_dir: The directory where the output HTML file will be saved.
        filename: The base filename for the output HTML file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create a proper timestamp column
    # Combine date and time columns into a single string
    time_str = evidence_df['date'].astype(str) + ' ' + evidence_df['time'].astype(str)
    
    # Convert to datetime objects, handling the AM/PM format
    evidence_df['timestamp'] = pd.to_datetime(time_str, format='%m/%d/%Y %I:%M:%S.%f %p')

    # Sort by timestamp to ensure the timeline is in order
    df = evidence_df.sort_values('timestamp').copy()

    # 2. Create the Gantt-style chart
    # Plotly's timeline requires a start and end time. For discrete events,
    # we'll make the end time slightly after the start time to create a small marker.
    df['timestamp_end'] = df['timestamp'] + pd.to_timedelta(1, unit='s')

    fig = px.timeline(
        df,
        x_start="timestamp",
        x_end="timestamp_end",
        y="problem_type",
        color="problem_type",
        hover_data=["sentence", "timestamp"],
        title=f"Event Log Timeline"
    )

    # Improve layout
    fig.update_yaxes(categoryorder='total ascending')
    fig.update_layout(
        title_font_size=22,
        font_size=14,
        xaxis_title="Timeline",
        yaxis_title="Problem Type",
        legend_title="Problem Types"
    )

    # 3. Save to HTML
    output_path = os.path.join(output_dir, f"chart_timeline.html")
    fig.write_html(output_path)
    print(f"Saved timeline chart to {output_path}")


def create_message_timeline(evidence_df: pd.DataFrame, output_dir: str):
    """
    Generates an HTML timeline of reconstructed messages with color-coded sentences based on problem type.

    Args:
        evidence_df: A Pandas DataFrame with 'message_id', 'timestamp', 'sentence', and 'problem_type'.
        output_dir: The directory where the output HTML file will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Define color mapping for problem types
    color_map = {
        'Normal': '#333333',  # Dark gray for normal text
        'SurroundingEnvironment': '#FF6347',  # Tomato
        'HardwareFault': '#DC143C',  # Crimson
        'ParamViolation': '#FF8C00',  # DarkOrange
        'RegulationViolation': '#9400D3',  # DarkViolet
        'CommunicationIssue': '#4169E1',  # RoyalBlue
        'SoftwareFault': "#F04351",  # ForestGreen
        'default': '#000000'
    }

    problem_map = {
        'Normal': 'N',
        'SurroundingEnvironment': 'SE',
        'HardwareFault': 'HF',
        'ParamViolation': 'PV',
        'RegulationViolation': 'RV',
        'CommunicationIssue': 'CI',
        'SoftwareFault': 'SF'
    }

    # 2. Prepare the data
    # Ensure sentences end with a period.
    evidence_df['sentence_punc'] = evidence_df['sentence'].apply(lambda s: s + '.' if not s.endswith('.') else s)
    
    # Ensure timestamp is present and sorted
    if 'timestamp' not in evidence_df.columns:
         time_str = evidence_df['date'].astype(str) + ' ' + evidence_df['time'].astype(str)
         evidence_df['timestamp'] = pd.to_datetime(time_str, format='%m/%d/%Y %I:%M:%S.%f %p')

    df = evidence_df.sort_values(['message_id', 'timestamp']).copy()

    # 3. Group by message_id and reconstruct messages with HTML styling
    timeline_entries = []
    for message_id, group in df.groupby('message_id'):
        # Get the timestamp from the first sentence of the message
        timestamp = group['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Reconstruct the message with colored sentences
        colored_message = ' '.join([
            f'<span style="color: {color_map.get(row["problem_type"], color_map["default"])}">[{problem_map.get(row['problem_type'])}] {row["sentence_punc"]}</span>'
            for _, row in group.iterrows()
        ])
        
        timeline_entries.append(f'<p><strong>{timestamp}</strong>: {colored_message}</p>')

    # 4. Generate the final HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Event Log Timeline</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; line-height: 1.6; }}
            h1 {{ color: #333; }}
            p {{ border-bottom: 1px solid #eee; padding-bottom: 10px; }}
            strong {{ color: #555; }}
        </style>
    </head>
    <body>
        <h1>Event Log Timeline</h1>
        {''.join(timeline_entries)}
    </body>
    </html>
    """

    # 5. Save to HTML file
    output_path = os.path.join(output_dir, "message_timeline.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Saved message timeline to {output_path}")