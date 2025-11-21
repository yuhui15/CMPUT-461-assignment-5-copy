"""
CMPUT 461 Assignment 3 - Task 3: Perplexity Evaluation

This file takes arguments to evaluate specific n-gram models
Returning the perplexity score of that model.

# Training Set Usage (5 tests)
python3 src/eval.py unigram data/training.json data/training.txt
python3 src/eval.py bigram data/training.json data/training.txt
python3 src/eval.py bigram data/training.json data/training.txt --laplace
python3 src/eval.py trigram data/training.json data/training.txt
python3 src/eval.py trigram data/training.json data/training.txt --laplace

# Dev Set Usage (5 tests)
python3 src/eval.py unigram data/training.json data/dev.txt
python3 src/eval.py bigram data/training.json data/dev.txt
python3 src/eval.py bigram data/training.json data/dev.txt --laplace
python3 src/eval.py trigram data/training.json data/dev.txt
python3 src/eval.py trigram data/training.json data/dev.txt --laplace
"""

import json
import sys
import math

# this is the function to evaluate the model accuracy
def evaluate_model(method, vocab_file, test_file, smooth):
    """
    Evalulate the n-gram model and compute perplexity
    """

    # Raise an exception if the command is invalid
    if method == 'unigram' and smooth == '--laplace':
        raise Exception("Please enter an valid command")
    if (method == 'unigram' or method == 'bigram' or method == 'trigram') == False:
        raise Exception("Please enter an valid command")
    if (smooth == None or smooth == '--laplace') == False:
        raise Exception("Please enter an valid command")

    # Load the vocab dictionary from f
    with open(vocab_file , "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Calculate vocabulary size (exclude <s> and </s>)
    vocab_size = len([w for w in vocab["unigrams"].keys() 
                if w not in ["<s>", "</s>"] and not w.startswith("[")])
    total_tokens = sum(count for word, count in vocab["unigrams"].items() if word not in ["<s>", "</s>"])

    N = 0
    perplexity_score = 0
    
    train_lines = []
    with open(test_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line: 
                continue
            tokens = line.split()
            tokens = [token if token in vocab["unigrams"] else '<unk>' for token in tokens]
            # Adding markers to match training 
            tokens = ["<s>"] + tokens + ["<s>"]
            train_lines.append(tokens)

    if method == 'unigram':
        for train_line in train_lines:
            for i in range(len(train_line)):
                # avoid first two tokens '<s>'
                if i == 0:
                    continue
                token = train_line[i]
                if token in vocab['unigrams']:
                    # apply the formula p(wi)/token_size\
                    perplexity_score += math.log(vocab['unigrams'][token] / total_tokens)
                    N += 1
                else:
                    # estimate the probability if the token is OOV
                    perplexity_score += math.log(1 / vocab_size)
                    N += 1
    
    # with the similar structure as unigram
    elif method == 'bigram':
        for train_line in train_lines:
            for i in range(len(train_line)-1):
                if i==0:
                    continue
                token_numerator = f"{train_line[i]}\t{train_line[i+1]}"
                token_denominator = train_line[i]
                if token_denominator in vocab['unigrams']:
                    denominator = vocab['unigrams'][token_denominator]
                else:
                    denominator = 0
                
                if token_numerator in vocab['bigrams']:
                    numerator = vocab['bigrams'][token_numerator]
                else:
                    numerator = 0
                
                # select the smooth function or not
                if smooth  == '--laplace':
                    numerator += 1
                    denominator += vocab_size
                    perplexity_score += math.log(numerator/denominator)
                    N += 1
                else:
                    if numerator == 0 or denominator == 0:
                        perplexity_score += math.log(1 / vocab_size)
                        N += 1
                    else:
                        perplexity_score += math.log(numerator/denominator)
                        N += 1
    # with the similar structure as unigram
    else:
        for train_line in train_lines:
            for i in range(len(train_line)-2):
                token_numerator = f"{train_line[i]}\t{train_line[i+1]}\t{train_line[i+2]}"
                token_denominator = f"{train_line[i]}\t{train_line[i+1]}"
                if token_denominator in vocab['bigrams']:
                    denominator = vocab['bigrams'][token_denominator]
                else:
                    denominator = 0
                
                if token_numerator in vocab['trigrams']:
                    numerator = vocab['trigrams'][token_numerator]
                else:
                    numerator = 0
                
                if smooth  == '--laplace':
                    numerator += 1
                    denominator += vocab_size
                    perplexity_score += math.log(numerator/denominator)
                    N += 1
                else:
                    if numerator == 0 or denominator == 0:
                        perplexity_score += math.log(1 / vocab_size)
                        N += 1
                    else:
                        perplexity_score += math.log(numerator/denominator)
                        N += 1
    
    # Perplexity Calculation
    perplexity_score = math.exp(- 1/N * perplexity_score)
    print(f" Perplexity Score: {perplexity_score}")


if __name__ == "__main__":
    method = sys.argv[1]
    vocab_file = sys.argv[2]
    test_file = sys.argv[3]
    smooth = sys.argv[4] if len(sys.argv) > 4 else None
    evaluate_model(method, vocab_file, test_file, smooth)