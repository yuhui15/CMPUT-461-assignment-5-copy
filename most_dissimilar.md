## Most dissimilar functionality

### Your code

```Python
Answer here.

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
```

### LLM code

```Python
Answer here.

# --------------------
# Data reading + OOV mapping
# --------------------

def read_utterances(path: str) -> List[List[str]]:
    utts: List[List[str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                utts.append(line.split())
    return utts


def map_oov(tokens: List[str], vocab: set[str]) -> List[str]:
    return [t if t in vocab else UNK for t in tokens]


# --------------------
# Load model JSON
# --------------------

def load_model(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)

    # Extract counts
    uni_counts: Dict[str, int] = data["unigram"]["counts"]

    bi_counts: Dict[Tuple[str, str], int] = {
        tuple(k.split()): v for k, v in data["bigram"]["counts"].items()
    }

    tri_counts: Dict[Tuple[str, str, str], int] = {
        tuple(k.split()): v for k, v in data["trigram"]["counts"].items()
    }

    meta = data["meta"]

    return {
        "meta": meta,
        "unigram": uni_counts,
        "bigram": bi_counts,
        "trigram": tri_counts
    }


# --------------------
# Probability estimators
# --------------------

def p_unigram(w: str, model: Dict[str, Any]) -> float:
    uni = model["unigram"]
    total = model["meta"]["unigram_total"]
    return uni.get(w, 0) / total


def p_bigram(w1: str, w2: str, model: Dict[str, Any], laplace: bool) -> float:
    bi = model["bigram"]
    uni = model["unigram"]
    V = model["meta"]["vocab_size_conditioning"]

    num = bi.get((w1, w2), 0)
    den = uni.get(w1, 0)

    if laplace:
        return (num + 1) / (den + V)

    if den == 0:
        return 0.0
    return num / den


def p_trigram(w1: str, w2: str, w3: str, model: Dict[str, Any], laplace: bool) -> float:
    tri = model["trigram"]
    bi = model["bigram"]
    V = model["meta"]["vocab_size_conditioning"]

    num = tri.get((w1, w2, w3), 0)
    den = bi.get((w1, w2), 0)

    if laplace:
        return (num + 1) / (den + V)

    if den == 0:
        return 0.0
    return num / den


# --------------------
# Perplexity computation
# --------------------

def compute_ppl(
    model_type: str,
    model: Dict[str, Any],
    utterances: List[List[str]],
    laplace: bool,
    vocab: set[str]
) -> float:

    log_prob_sum = 0.0
    N = 0

    for utt in utterances:
        utt = map_oov(utt, vocab)
        seq = [START] + utt + [END]

        if model_type == "unigram":
            for w in seq:
                p = p_unigram(w, model)
                if p == 0:
                    return float("inf")
                log_prob_sum += math.log(p)
                N += 1

        elif model_type == "bigram":
            for i in range(len(seq) - 1):
                p = p_bigram(seq[i], seq[i + 1], model, laplace)
                if p == 0:
                    return float("inf")
                log_prob_sum += math.log(p)
                N += 1

        elif model_type == "trigram":
            for i in range(len(seq) - 2):
                p = p_trigram(seq[i], seq[i + 1], seq[i + 2], model, laplace)
                if p == 0:
                    return float("inf")
                log_prob_sum += math.log(p)
                N += 1

    return math.exp(-log_prob_sum / N)
```

### Output of `diff`

```
Answer here. 
Create two temp files (no need to commit them) f1.py (your code) and f2.py (llm code); run diff f1.py f2.py and copy/paste the output here

$ diff f1.py f2.py
1,22c1,3
< # this is the function to evaluate the model accuracy
< def evaluate_model(method, vocab_file, test_file, smooth):
<     """
<     Evalulate the n-gram model and compute perplexity
<     """
<
<     # Raise an exception if the command is invalid
<     if method == 'unigram' and smooth == '--laplace':
<         raise Exception("Please enter an valid command")
<     if (method == 'unigram' or method == 'bigram' or method == 'trigram') == False:
<         raise Exception("Please enter an valid command")
<     if (smooth == None or smooth == '--laplace') == False:
<         raise Exception("Please enter an valid command")
<
<     # Load the vocab dictionary from f
<     with open(vocab_file , "r", encoding="utf-8") as f:
<         vocab = json.load(f)
<
<     # Calculate vocabulary size (exclude <s> and </s>)
<     vocab_size = len([w for w in vocab["unigrams"].keys()
<                 if w not in ["<s>", "</s>"] and not w.startswith("[")])
<     total_tokens = sum(count for word, count in vocab["unigrams"].items() if word not in ["<s>", "</s>"])
---
> # --------------------
> # Data reading + OOV mapping
> # --------------------
24,30c5,8
<     N = 0
<     perplexity_score = 0
<
<     train_lines = []
<     with open(test_file, "r", encoding="utf-8") as f:
<         lines = f.readlines()
<         for line in lines:
---
> def read_utterances(path: str) -> List[List[str]]:
>     utts: List[List[str]] = []
>     with open(path, "r") as f:
>         for line in f:
32,118c10,133
<             if not line:
<                 continue
<             tokens = line.split()
<             tokens = [token if token in vocab["unigrams"] else '<unk>' for token in tokens]
<             # Adding markers to match training
<             tokens = ["<s>"] + tokens + ["<s>"]
<             train_lines.append(tokens)
<
<     if method == 'unigram':
<         for train_line in train_lines:
<             for i in range(len(train_line)):
<                 # avoid first two tokens '<s>'
<                 if i == 0:
<                     continue
<                 token = train_line[i]
<                 if token in vocab['unigrams']:
<                     # apply the formula p(wi)/token_size\
<                     perplexity_score += math.log(vocab['unigrams'][token] / total_tokens)
<                     N += 1
<                 else:
<                     # estimate the probability if the token is OOV
<                     perplexity_score += math.log(1 / vocab_size)
<                     N += 1
<
<     # with the similar structure as unigram
<     elif method == 'bigram':
<         for train_line in train_lines:
<             for i in range(len(train_line)-1):
<                 if i==0:
<                     continue
<                 token_numerator = f"{train_line[i]}\t{train_line[i+1]}"
<                 token_denominator = train_line[i]
<                 if token_denominator in vocab['unigrams']:
<                     denominator = vocab['unigrams'][token_denominator]
<                 else:
<                     denominator = 0
<
<                 if token_numerator in vocab['bigrams']:
<                     numerator = vocab['bigrams'][token_numerator]
<                 else:
<                     numerator = 0
<
<                 # select the smooth function or not
<                 if smooth  == '--laplace':
<                     numerator += 1
<                     denominator += vocab_size
<                     perplexity_score += math.log(numerator/denominator)
<                     N += 1
<                 else:
<                     if numerator == 0 or denominator == 0:
<                         perplexity_score += math.log(1 / vocab_size)
<                         N += 1
<                     else:
<                         perplexity_score += math.log(numerator/denominator)
<                         N += 1
<     # with the similar structure as unigram
<     else:
<         for train_line in train_lines:
<             for i in range(len(train_line)-2):
<                 token_numerator = f"{train_line[i]}\t{train_line[i+1]}\t{train_line[i+2]}"
<                 token_denominator = f"{train_line[i]}\t{train_line[i+1]}"
<                 if token_denominator in vocab['bigrams']:
<                     denominator = vocab['bigrams'][token_denominator]
<                 else:
<                     denominator = 0
<
<                 if token_numerator in vocab['trigrams']:
<                     numerator = vocab['trigrams'][token_numerator]
<                 else:
<                     numerator = 0
<
<                 if smooth  == '--laplace':
<                     numerator += 1
<                     denominator += vocab_size
<                     perplexity_score += math.log(numerator/denominator)
<                     N += 1
<                 else:
<                     if numerator == 0 or denominator == 0:
<                         perplexity_score += math.log(1 / vocab_size)
<                         N += 1
<                     else:
<                         perplexity_score += math.log(numerator/denominator)
<                         N += 1
<
<     # Perplexity Calculation
<     perplexity_score = math.exp(- 1/N * perplexity_score)
<     print(f" Perplexity Score: {perplexity_score}")
\ No newline at end of file
---
>             if line:
>                 utts.append(line.split())
>     return utts
>
>
> def map_oov(tokens: List[str], vocab: set[str]) -> List[str]:
>     return [t if t in vocab else UNK for t in tokens]
>
>
> # --------------------
> # Load model JSON
> # --------------------
>
> def load_model(path: str) -> Dict[str, Any]:
>     with open(path, "r") as f:
>         data = json.load(f)
>
>     # Extract counts
>     uni_counts: Dict[str, int] = data["unigram"]["counts"]
>
>     bi_counts: Dict[Tuple[str, str], int] = {
>         tuple(k.split()): v for k, v in data["bigram"]["counts"].items()
>     }
>
>     tri_counts: Dict[Tuple[str, str, str], int] = {
>         tuple(k.split()): v for k, v in data["trigram"]["counts"].items()
>     }
>
>     meta = data["meta"]
>
>     return {
>         "meta": meta,
>         "unigram": uni_counts,
>         "bigram": bi_counts,
>         "trigram": tri_counts
>     }
>
>
> # --------------------
> # Probability estimators
> # --------------------
>
> def p_unigram(w: str, model: Dict[str, Any]) -> float:
>     uni = model["unigram"]
>     total = model["meta"]["unigram_total"]
>     return uni.get(w, 0) / total
>
>
> def p_bigram(w1: str, w2: str, model: Dict[str, Any], laplace: bool) -> float:
>     bi = model["bigram"]
>     uni = model["unigram"]
>     V = model["meta"]["vocab_size_conditioning"]
>
>     num = bi.get((w1, w2), 0)
>     den = uni.get(w1, 0)
>
>     if laplace:
>         return (num + 1) / (den + V)
>
>     if den == 0:
>         return 0.0
>     return num / den
>
>
> def p_trigram(w1: str, w2: str, w3: str, model: Dict[str, Any], laplace: bool) -> float:
>     tri = model["trigram"]
>     bi = model["bigram"]
>     V = model["meta"]["vocab_size_conditioning"]
>
>     num = tri.get((w1, w2, w3), 0)
>     den = bi.get((w1, w2), 0)
>
>     if laplace:
>         return (num + 1) / (den + V)
>
>     if den == 0:
>         return 0.0
>     return num / den
>
>
> # --------------------
> # Perplexity computation
> # --------------------
>
> def compute_ppl(
>     model_type: str,
>     model: Dict[str, Any],
>     utterances: List[List[str]],
>     laplace: bool,
>     vocab: set[str]
> ) -> float:
>
>     log_prob_sum = 0.0
>     N = 0
>
>     for utt in utterances:
>         utt = map_oov(utt, vocab)
>         seq = [START] + utt + [END]
>
>         if model_type == "unigram":
>             for w in seq:
>                 p = p_unigram(w, model)
>                 if p == 0:
>                     return float("inf")
>                 log_prob_sum += math.log(p)
>                 N += 1
>
>         elif model_type == "bigram":
>             for i in range(len(seq) - 1):
>                 p = p_bigram(seq[i], seq[i + 1], model, laplace)
>                 if p == 0:
>                     return float("inf")
>                 log_prob_sum += math.log(p)
>                 N += 1
>
>         elif model_type == "trigram":
>             for i in range(len(seq) - 2):
>                 p = p_trigram(seq[i], seq[i + 1], seq[i + 2], model, laplace)
>                 if p == 0:
>                     return float("inf")
>                 log_prob_sum += math.log(p)
>                 N += 1
>
>     return math.exp(-log_prob_sum / N)
\ No newline at end of file

```

## Discussion

### What is most similar?

```
Answer here.
The logic? The order of steps? The style? Variable names?
What could explain the similarity?
The most similar aspect is the logic of the n-gram training pipeline.
(1) read the utterance from the input file
(2) replace the OOV token by the <unk> label
(3) count the probability of each token based on the n-gram model
(4) compute the perplexity score
My code and chatgpt code all follow the pipeline above, so the order of steps should also be the same.
The variable names are similar. For example, 'vocab["unigrams"]', 'vocab['bigrams']' and 'vocab['trigrams']' variables are similar to 'uni', 'bi', 'tri'.
```

### What is most different?

```
Answer here.
The logic? The order of steps? The style? Variable names?
What could explain the dissimilarity?
The most different aspect is the number of functions and the size of each module.
In my code, there is only one function evaluate_model to evaluate the model.
However, there are several functions containing read_utterances, map_oov, load_model, p_unigram, p_bigram, p_trigram, compute_ppl to evaluate the model. Therefore, we find that the size of each module in my code is larger than one in chatgpt code.
```
