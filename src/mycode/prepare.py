"""
CMPUT 461 Assignment 3 - Task 1: Data Preparation

1. Processes raw .cha files and cleans them (from Assignment 1)
2. Transforms cleaned text to phonetic sequences using CMU dict
3. Consolidates all phonetic sequences
4. Splits them into training.txt (80%) and dev.txt (20%)

Usage: python3 src/prepare_data.py
"""

import os
import re
import random
import time

random.seed(42)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CMU_PATH = os.path.join(ROOT_DIR, "cmudict-0.7b")

# ============================================================
# FILE FINDING AND CLEANING
# ============================================================

def find_cha_files(root_dir):
    files = []
    for dir_path, _, file_names in os.walk(root_dir):
        for fname in file_names:
            if fname.endswith(".cha"):
                files.append(os.path.join(dir_path, fname))
    return files


def clean_utterance(text):
    """
    Clean utterance by removing CHAT format markers and irrelevant content.
    """

    # Remove spans enclosed by the same non-word, non-space delimiter
    text = re.sub(r'([^\w\s])(?:(?!\n).)*?\1', '', text)

    # Remove speaker sounds / events starting with &=, &~, &-
    text = re.sub(r'&[=~-].*?(?=\s|$)', '', text)

    # Replacement "orig [: repair]" -> "repair"
    text = re.sub(r'\b[^\s]+\s*\[:\s*(.*?)\]', r'\1', text)

    # Remove denoted repetitions/corrections with [//] or [/]
    text = re.sub(r'[^\s]+\s*\[\/\/?\]', '', text)

    # Unwrap angle-bracket scope that has trailing bracketed scope marks
    text = re.sub(r'<(.*?)>\s*\[.*?\]', r'\1', text)

    # Remove brackets that trap words
    text = re.sub(r'\[[^\w]*\s*([\w+\s]+)\]', r'\1', text)
    
    # Remove special markers and placeholder words
    text = re.sub(r'(<&~s>|xxx|yyy|www|iiiib)', '', text, flags=re.IGNORECASE)
    
    # Remove @l suffix to keep letter
    text = re.sub(r"([a-zA-Z])@l", r"\1", text)

    # Combine colon-joined "extended" words: long:word -> longword
    text = re.sub(r'(\w+):(\w+)', r'\1\2', text)
    
    # Replace underscores between letters with space
    text = re.sub(r'(?<=\w)_(?=\w)', ' ', text)
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove irrelevant punctuation (keep apostrophes, periods, ?, !)
    text = re.sub(r"[^\w\s'\.\?!\n]", "", text)
    
    # Remove lines with only punctuation
    text = re.sub(r'(?m)^\s*[.!?]+\s*\n?', '', text)
    
    # Clean whitespace
    text = "\n".join(re.sub(r"\s+", " ", line).strip() 
                     for line in text.splitlines() if line.strip())
    
    return text


def extract_clean_utterances(file_path):
    """
    Extract and clean utterances from a .cha file.
    Returns list of cleaned utterance strings.
    """
    cleaned_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, headers, and comments
            if not line or line.startswith("@") or line.startswith("%"):
                continue
            # Process speaker lines (start with *)
            if line.startswith("*"):
                line = line.split(":", 1)[-1].strip()
                line = clean_utterance(line)
                if line:
                    cleaned_lines.append(line)
    return cleaned_lines


# ============================================================
# CMU DICTIONARY AND PHONEME TRANSFORMATION
# ============================================================

def load_cmu_dict(path):
    """
    Load CMU pronunciation dictionary into a hashmap.
    Returns dict (key: word, value: phoneme sequence)
    """
    cmu_dict = {}
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            parts = line.split()
            word = parts[0]
            phonemes = parts[1:]
            base_word = word.split('(')[0] 
            cmu_dict.setdefault(base_word, []).append(phonemes)
    return cmu_dict


phoneme_cache = {}
def lookup_phonemes(word, cmu_dict):
    """
    Look up phonemes for a word in CMU dictionary.
    Returns list of phonemes, or [WORD] if not found.
    """
    cache_key = (word)
    if cache_key in phoneme_cache:
        return phoneme_cache[cache_key]
    
    def get_from_cmu(token):
        if token in cmu_dict:
            phonemes = cmu_dict[token][0]  # Use first pronunciation
            return phonemes
    
    phonemes = get_from_cmu(word)
    
    # Special case: handle possessive 's
    if phonemes is None and word.endswith("'S") and word != "'S":
        base = word[:-2]
        base_phonemes = get_from_cmu(base)
        s_phonemes = get_from_cmu("'S")
        if base_phonemes is not None and s_phonemes is not None:
            phonemes = base_phonemes + s_phonemes
    
    # If still not found, wrap in brackets
    if phonemes is None:
        phonemes = [f"[{word}]"]
    
    phoneme_cache[cache_key] = phonemes
    return phonemes


def text_to_phonemes(text, cmu_dict, keep_stress=False):
    """
    Convert text utterance to phoneme sequence.
    Returns list of phoneme strings.
    """
    words = re.findall(r"[A-Za-z']+", text.upper())
    phoneme_sequence = []
    
    for word in words:
        phonemes = lookup_phonemes(word, cmu_dict)
        phoneme_sequence.extend(phonemes)
    
    return phoneme_sequence


# ============================================================
# DATA CONSOLIDATION AND RANDOM DIVIDING
# ============================================================  

def split_data(utterances, train_ratio=0.8):
    """
    Randomly split utterances into training and dev sets.
    Returns tuple of (training_list, dev_list).
    """
    random.shuffle(utterances)
    split_index = int(len(utterances) * train_ratio)
    return utterances[:split_index], utterances[split_index:]


def save_to_file(utterances, file_path):
    """
    Save utterances to file, one per line.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for utterance in utterances:
            f.write(utterance + '\n')

# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    start_time = time.time()
    
    print("=" * 60)
    print("CMPUT 461 Assignment 3: Data Preparation Pipeline")
    print("=" * 60)
    
    # Step 1: Initalization
    print("\n[Step 1] Initalization of Data")
    cmu_dict = load_cmu_dict(CMU_PATH)
    print(f">> Loaded {len(cmu_dict)} words from CMU dictionary")

    cha_files = find_cha_files(RAW_DIR)
    print(f">> Found {len(cha_files)} .cha files")
    
    # Step 2: Cleaning and Transformation Process
    print("\n[Step 2] Processing Files & Transforming to Phonemes")
    all_phoneme_utterances = []
    
    for i, file_path in enumerate(cha_files, 1):
        if i % 20 == 0:
            print(f">> Processing file {i}/{len(cha_files)}...")
        
        # Extract and clean utterances
        cleaned_utterances = extract_clean_utterances(file_path)
        
        # Transform each utterance to phonemes
        for utterance in cleaned_utterances:
            phonemes = text_to_phonemes(utterance, cmu_dict, keep_stress=False)
            if phonemes:  # Only add non-empty sequences
                phoneme_string = " ".join(phonemes)
                all_phoneme_utterances.append(phoneme_string)
    
    print(f">> Total utterances collected: {len(all_phoneme_utterances)}")

    # Step 3: Split into training and dev sets
    print("\n[Step 3] Splitting Data")
    training, dev = split_data(all_phoneme_utterances, train_ratio=0.8)
    print(f">> Training set: {len(training)} utterances ({len(training)/len(all_phoneme_utterances)*100:.1f}%)")
    print(f">> Dev set: {len(dev)} utterances ({len(dev)/len(all_phoneme_utterances)*100:.1f}%)")
    
    # Step 4: Save to their respective files
    print("\n[Step 4] Preparing Files")
    training_path = os.path.join(DATA_DIR, "training.txt")
    dev_path = os.path.join(DATA_DIR, "dev.txt")
    
    save_to_file(training, training_path)
    save_to_file(dev, dev_path)
    
    print(f">> Training set saved to: {training_path}")
    print(f">> Dev set saved to: {dev_path}")
    
    # Print Preparation Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f">> Processed {len(cha_files)} files in {total_time:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()