## Most similar functionality

### Your code

```Python
Answer here.

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
```

### LLM code

```Python
Answer here.

def read_utterances(path: str) -> List[List[str]]:
    """Reads a data file. Each line is one space-separated utterance."""
    utts: List[List[str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                utts.append(line.split())
    return utts


# --------------------
# OOV handling
# --------------------

def build_vocab(utterances: List[List[str]]) -> List[str]:
    """Singletons → <unk> as per J&M Chapter 3."""
    counts = Counter()

    for utt in utterances:
        counts.update(utt)

    vocab = [tok for tok, c in counts.items() if c > 1]
    vocab.append(UNK)
    return vocab


def replace_singletons_with_unk(
    utterances: List[List[str]],
    vocab: List[str]
) -> List[List[str]]:
    vocab_set = set(vocab)
    return [
        [tok if tok in vocab_set else UNK for tok in utt]
        for utt in utterances
    ]


# --------------------
# N-gram counting
# --------------------

def count_ngrams(
    utterances: List[List[str]]
) -> Tuple[
    Counter[str],
    Counter[Tuple[str, str]],
    Counter[Tuple[str, str, str]]
]:
    uni: Counter[str] = Counter()
    bi: Counter[Tuple[str, str]] = Counter()
    tri: Counter[Tuple[str, str, str]] = Counter()

    for utt in utterances:
        seq = [START] + utt + [END]

        uni.update(seq)

        for i in range(len(seq) - 1):
            bi[(seq[i], seq[i + 1])] += 1

        for i in range(len(seq) - 2):
            tri[(seq[i], seq[i + 1], seq[i + 2])] += 1

    return uni, bi, tri


# --------------------
# JSON formatting
# --------------------

def convert_to_json(counter: Counter) -> Dict[str, int]:
    """Convert tuple keys → 'w1 w2' strings."""
    out: Dict[str, int] = {}
    for k, v in counter.items():
        if isinstance(k, tuple):
            out[" ".join(k)] = v
        else:
            out[k] = v
    return out


def create_model_json(
    uni: Counter[str],
    bi: Counter[Tuple[str, str]],
    tri: Counter[Tuple[str, str, str]],
    vocab: List[str],
    n_utterances: int
) -> Dict[str, Any]:

    V = len(vocab)

    return {
        "meta": {
            "description": "n-gram counts without smoothing",
            "bos": START,
            "eos": END,
            "unk": UNK,
            "n_sentences": n_utterances,
            "unigram_total": sum(uni.values()),
            "vocab_size_conditioning": V
        },
        "unigram": {
            "order": 1,
            "counts": convert_to_json(uni)
        },
        "bigram": {
            "order": 2,
            "counts": convert_to_json(bi)
        },
        "trigram": {
            "order": 3,
            "counts": convert_to_json(tri)
        }
    }
```

### Output of `diff`

```
Answer here. 
Create two temp files (no need to commit them) f1.py (your code) and f2.py (llm code); run diff f1.py f2.py and copy/paste the output here

$ diff -u f1.py f2.py
--- f1.py       2025-11-20 20:25:45.325944700 -0700
+++ f2.py       2025-11-20 20:26:25.742979300 -0700
@@ -1,80 +1,115 @@
-def train_model(input_file, output_file):
-    """
-    Train unigram, bigram, and trigram models from phoneme sequences
-    Saves the n-gram counts into a JSON file,
-    """
-
-    unigrams_dict = {}
-    bigrams_dict = {}
-    trigrams_dict = {}
-    frequency_dict = {}
-    print("=" * 60)
-    print("CMPUT 461 Assignment 3: Data Training Pipeline ")
-    print("=" * 60)
-
-    with open(input_file , "r", encoding="utf-8") as f:
-        lines = f.readlines()
-
-    print(f"[Step 1] Processing {len(lines)} utterances")
-
-    for line in lines:
-        line = line.strip()
-        tokens = line.split() # Phoneme sequences need to be split
-        if len(tokens) == 0:
-            continue
-        # count the frequency for each token
-        for token in tokens:
-            if token in frequency_dict:
-                frequency_dict[token] += 1
-            else:
-                frequency_dict[token] = 1
-
-    for line in lines:
-        line = line.strip()
-        if not line:
-            continue
-
-        tokens = line.split() # Phoneme sequences need to be split
-        tokens = [token if frequency_dict[token] > UNK_THRESHOLD else '<unk>' for token in tokens]
-
-        # Unigram counting
-        for token in tokens:
-            unigrams_dict[token] = unigrams_dict.get(token, 0) + 1
-
-        # add first layer of start / end of sentence tokens
-        tokens = ["<s>"] + tokens + ["</s>"]
-        # Bigram counting
-        for i in range(len(tokens) - 1):
-            bigram = f"{tokens[i]}\t{tokens[i+1]}"
-            bigrams_dict[bigram] = bigrams_dict.get(bigram, 0) + 1
-
-        # add second layer of start / end of sentence tokens
-        tokens = ["<s>"] + tokens + ["</s>"]
-        # Trigrams counting
-        for i in range(len(tokens) - 2):
-            trigram = f"{tokens[i]}\t{tokens[i+1]}\t{tokens[i+2]}"
-            trigrams_dict[trigram] = trigrams_dict.get(trigram, 0) + 1
-
-
-
-
-    # Create model with all n-gram counts
-    model = {
-        "unigrams": unigrams_dict,
-        "bigrams": bigrams_dict,
-        "trigrams": trigrams_dict,
+def read_utterances(path: str) -> List[List[str]]:
+    """Reads a data file. Each line is one space-separated utterance."""
+    utts: List[List[str]] = []
+    with open(path, "r") as f:
+        for line in f:
+            line = line.strip()
+            if line:
+                utts.append(line.split())
+    return utts
+
+
+# --------------------
+# OOV handling
+# --------------------
+
+def build_vocab(utterances: List[List[str]]) -> List[str]:
+    """Singletons → <unk> as per J&M Chapter 3."""
+    counts = Counter()
+
+    for utt in utterances:
+        counts.update(utt)
+
+    vocab = [tok for tok, c in counts.items() if c > 1]
+    vocab.append(UNK)
+    return vocab
+
+
+def replace_singletons_with_unk(
+    utterances: List[List[str]],
+    vocab: List[str]
+) -> List[List[str]]:
+    vocab_set = set(vocab)
+    return [
+        [tok if tok in vocab_set else UNK for tok in utt]
+        for utt in utterances
+    ]
+
+
+# --------------------
+# N-gram counting
+# --------------------
+
+def count_ngrams(
+    utterances: List[List[str]]
+) -> Tuple[
+    Counter[str],
+    Counter[Tuple[str, str]],
+    Counter[Tuple[str, str, str]]
+]:
+    uni: Counter[str] = Counter()
+    bi: Counter[Tuple[str, str]] = Counter()
+    tri: Counter[Tuple[str, str, str]] = Counter()
+
+    for utt in utterances:
+        seq = [START] + utt + [END]
+
+        uni.update(seq)
+
+        for i in range(len(seq) - 1):
+            bi[(seq[i], seq[i + 1])] += 1
+
+        for i in range(len(seq) - 2):
+            tri[(seq[i], seq[i + 1], seq[i + 2])] += 1
+
+    return uni, bi, tri
+
+
+# --------------------
+# JSON formatting
+# --------------------
+
+def convert_to_json(counter: Counter) -> Dict[str, int]:
+    """Convert tuple keys → 'w1 w2' strings."""
+    out: Dict[str, int] = {}
+    for k, v in counter.items():
+        if isinstance(k, tuple):
+            out[" ".join(k)] = v
+        else:
+            out[k] = v
+    return out
+
+
+def create_model_json(
+    uni: Counter[str],
+    bi: Counter[Tuple[str, str]],
+    tri: Counter[Tuple[str, str, str]],
+    vocab: List[str],
+    n_utterances: int
+) -> Dict[str, Any]:
+
+    V = len(vocab)
+
+    return {
+        "meta": {
+            "description": "n-gram counts without smoothing",
+            "bos": START,
+            "eos": END,
+            "unk": UNK,
+            "n_sentences": n_utterances,
+            "unigram_total": sum(uni.values()),
+            "vocab_size_conditioning": V
+        },
+        "unigram": {
+            "order": 1,
+            "counts": convert_to_json(uni)
+        },
+        "bigram": {
+            "order": 2,
+            "counts": convert_to_json(bi)
+        },
+        "trigram": {
+            "order": 3,
+            "counts": convert_to_json(tri)
+        }
     }
-
-    # Write model to JSON file
-    print(f"[Step 2] Saving model to output file: {output_file}")
-    with open(output_file, 'w', encoding='utf-8') as out:
-        json.dump(model, out, ensure_ascii=False, indent=2)
-
-    print("=" * 60)
-    print("Model Statistics:")
-    print(f">> Unique unigrams: {len(unigrams_dict)}")
-    print(f">> Unique bigrams: {len(bigrams_dict)}")
-    print(f">> Unique trigrams: {len(trigrams_dict)}")
-    print(f">> Total tokens: {sum(unigrams_dict.values())}")
-    print("=" * 60)
-    print(f"Model Saved Successfully.")

```

## Discussion

### What is most similar?

```
Answer here.
The logic? The order of steps? The style? Variable names?
What could explain the similarity?
The most similar aspect is the logic of the n-gram training pipeline.
(1) read the utterance from the input file
(2) replace the low frequency token by the <unk> label
(3) count unigram, bigram and trigram
(4) put the model into the JSON model
My code and chatgpt code all follow the pipeline above, so the order of steps should also be the same.
The variable names are similar. For example, 'unigrams_dict', 'bigrams_dict' and 'trigrams_dict' variables are similar to 'uni', 'bi', 'tri'.
```

### What is most different?

```
Answer here.
The logic? The order of steps? The style? Variable names?
What could explain the dissimilarity?
The most different aspect is the number of functions and the size of each module.
In my code, there is only one function train_model to train the model.
However, there are several functions containing read_utterances, build_vocab, replace_singletons_with_unk, count_ngrams, convert_to_json, create_model_json to train the model.Therefore, we find that the size of each module in my code is larger than one in chatgpt code.
```
