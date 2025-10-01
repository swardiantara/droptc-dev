import re
import os
import pandas as pd

from src.cli.utils import extract_message

def _segment_by_colon(text: str) -> list[str]:
    """
    Helper function to segment text by colons based on the word count rule.
    A split occurs only if the resulting left and right parts both contain at least two words.
    """
    # Don't process if there's no colon to split by
    if ':' not in text:
        return [text]

    # Split the text by colons that are NOT part of a time format (e.g., 14:30),
    # but keep the colon itself in the resulting list to decide if we merge or split.
    # The regex `((?<!\d):(?!\d))` finds a colon not surrounded by digits and captures it.
    parts = re.split(r'((?<!\d):(?!\d))', text)
    
    # If the text starts or ends with a separator, re.split can create empty strings.
    # Filter them out before processing.
    parts = [p for p in parts if p]

    if not parts:
        return []

    # Start with the first chunk of text as our first sentence
    segmented_list = [parts[0]]
    
    # Iterate through the parts list, starting from the first potential delimiter
    i = 1
    while i < len(parts):
        delimiter = parts[i]
        text_chunk_after = parts[i+1]
        
        # Count words in the text before this colon (the last sentence we have)
        # and the text after this colon (the next chunk in our list).
        words_before = len(segmented_list[-1].strip().split())
        words_after = len(text_chunk_after.strip().split())
        # Rule 3: Check if both sides have at least two words
        if words_before >= 2 and words_after >= 2 and not ('code' in text_chunk_after or 'Code' in text_chunk_after):
            # If so, this is a valid sentence break. Start a new sentence.
            segmented_list.append(text_chunk_after)
        else:
            # Otherwise, merge the colon and the next text chunk back
            # into the current last sentence.
            segmented_list[-1] += delimiter + text_chunk_after
            
        i += 2  # Advance past the delimiter and the text chunk
        
    return segmented_list

def segment_message(record: str) -> list[str]:
    """
    Segments a drone log record into sentences based on a specific set of rules.

    Args:
        record: A string containing the multi-sentence log message.

    Returns:
        A list of strings, where each string is a segmented sentence.
    """
    # --- 1. Quote Removal Preprocessing Step ---
    # Remove all double quotes, including straight ("") and smart (“”) variations.
    if '"' in record:
        record = record.replace('"', '')
    # record = re.sub(r'["“”]', '', record)
    
    # Remove single quotes, including straight (') and smart (‘’) variations.
    # This regex protects apostrophes, including possessive plurals like "students'".
    record = re.sub(r"(?<!\w)['’‘]|(?<!s)['’‘](?!\w)", '', record)
    
    # --- 2. Sentence Segmentation ---
    # Rule 1 & 2: Use period and comma as separators, unless part of a number.
    temp_record = re.sub(r'(?<!\d)[.;!](?!\d)', '|||', record)
    
    preliminary_sentences = temp_record.split('|||')
    
    # Rule 3: Process each preliminary sentence for the colon rule
    final_sentences = []
    for sentence in preliminary_sentences:
        colon_segmented_sentences = _segment_by_colon(sentence)
        final_sentences.extend(colon_segmented_sentences)
        
    # --- 3. Final Cleanup ---
    cleaned_sentences = [s.strip() for s in final_sentences if s.strip()]
    
    return cleaned_sentences


def segment_evidence(evidence_file: str, output_dir: str) -> pd.DataFrame:
    """
    Segments log messages from a CSV evidence file and saves the segmented output to an Excel file.

    Args:
        evidence_file: Path to the CSV file containing the evidence data.
        output_dir: The path to the 'output_dir'.
    Returns:
        A Pandas DataFrame containing the segmented log messages with additional metadata.
    """

    # Extract the base filename without extension for naming the output file
    base_filename = os.path.splitext(os.path.basename(evidence_file))[0]
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract messages from the evidence file
    messages_df = extract_message(evidence_file)
    
    # Initialize lists to hold the segmented data
    dates = []
    times = []
    message_ids = []
    messages = []
    sentences = []
    
    # Segment each message and populate the lists
    for idx, row in messages_df.iterrows():
        message_id = idx + 1  # Message IDs start from 1
        date = row['date']
        time = row['time']
        message = row['message']
        
        segmented_sentences = segment_message(message)
        
        for sentence in segmented_sentences:
            dates.append(date)
            times.append(time)
            message_ids.append(message_id)
            messages.append(message)
            sentences.append(sentence)
    
    # Create a DataFrame from the segmented data
    segmented_df = pd.DataFrame({
        'date': dates,
        'time': times,
        'message_id': message_ids,
        'message': messages,
        'sentence': sentences
    })
    
    # Save the segmented DataFrame to an Excel file
    output_filepath = os.path.join(output_dir, f'parsed_{base_filename}.xlsx')
    segmented_df.to_excel(output_filepath, index=False)
    
    print(f"Segmented log messages saved to {output_filepath}")
    
    return segmented_df