"""
CMPUT 461 Assignment 3 - Task 2: Training N-gram Models

This file trains unigram, bigram, and trigram models from phoneme sequences.

Usage: python3 src/train.py <input_file> <output_file>
>> python3 src/train.py data/training.txt data/training.json
"""

import json
import sys

UNK_THRESHOLD = 20

def train_model(input_file, output_file):
    """
    Train unigram, bigram, and trigram models from phoneme sequences 
    Saves the n-gram counts into a JSON file, 
    """

    unigrams_dict = {}
    bigrams_dict = {}
    trigrams_dict = {}
    frequency_dict = {}
    print("=" * 60)
    print("CMPUT 461 Assignment 3: Data Training Pipeline ")
    print("=" * 60)

    with open(input_file , "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    print(f"[Step 1] Processing {len(lines)} utterances")

    for line in lines:
        line = line.strip()
        tokens = line.split() # Phoneme sequences need to be split 
        if len(tokens) == 0:
            continue
        # count the frequency for each token
        for token in tokens:
            if token in frequency_dict:
                frequency_dict[token] += 1
            else:
                frequency_dict[token] = 1

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        tokens = line.split() # Phoneme sequences need to be split 
        tokens = [token if frequency_dict[token] > UNK_THRESHOLD else '<unk>' for token in tokens]

        # Unigram counting 
        for token in tokens:
            unigrams_dict[token] = unigrams_dict.get(token, 0) + 1
        
        # add first layer of start / end of sentence tokens
        tokens = ["<s>"] + tokens + ["</s>"]
        # Bigram counting
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}\t{tokens[i+1]}"
            bigrams_dict[bigram] = bigrams_dict.get(bigram, 0) + 1

        # add second layer of start / end of sentence tokens
        tokens = ["<s>"] + tokens + ["</s>"]
        # Trigrams counting
        for i in range(len(tokens) - 2):
            trigram = f"{tokens[i]}\t{tokens[i+1]}\t{tokens[i+2]}"
            trigrams_dict[trigram] = trigrams_dict.get(trigram, 0) + 1


                

    # Create model with all n-gram counts
    model = {
        "unigrams": unigrams_dict,
        "bigrams": bigrams_dict,
        "trigrams": trigrams_dict,
    }

    # Write model to JSON file
    print(f"[Step 2] Saving model to output file: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(model, out, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Model Statistics:")
    print(f">> Unique unigrams: {len(unigrams_dict)}")
    print(f">> Unique bigrams: {len(bigrams_dict)}")
    print(f">> Unique trigrams: {len(trigrams_dict)}")
    print(f">> Total tokens: {sum(unigrams_dict.values())}")
    print("=" * 60)
    print(f"Model Saved Successfully.")

if __name__ == "__main__":

    # Invalid Argument Catch
    if len(sys.argv) != 3:
        print("Usage Requirements: python3 src/train.py <input_file> <output_file>")
        sys.exit(1)

    train_file = sys.argv[1]
    output_file = sys.argv[2]
    train_model(train_file, output_file)