# Answer these questions with respect to the data in the `training` set.

## Does the LLM handle begin and end of sentence tokens across models correctly?

```
Answer here.
No. The LLM doesn't handle begin and end of sentence tokens across models correctly.
Explain (with examples if needed) why or why not.
The code only turns the utterance into [<s>] + utterance + [</s>] for unigram, bigram, trigram models.
The correct method is to turn the utterance into utterance, [<s>] + utterance + [</s>], and [<s>, <s>] + utterance + [</s>, </s>] for unigram, bigram and trigram models respectively. If we use LLM code, we cannot compute the count of the first token of the utterance for trigram model. 
```

### Does your code handle these tokens in the same way?

```
Answer here.
No.
If the answer is no, explain why your answer is better or worse.
My answer is better since my method is to turn the utterance into utterance, [<s>] + utterance + [</s>], and [<s>, <s>] + utterance + [</s>, </s>] for unigram, bigram and trigram models respectively. Therefore, we can compute the count of the first token of the utterance for trigram model.
```

----


## Does the LLM count n-grams correctly?

```
Answer here.
Explain (with examples if needed) why or why not.
HINT: use the tiny models.
I have manually check the tiny models trained on tiny_training.txt by searching the n-grams in tiny_training.txt
The following is the content of tiny_training.txt
HH EH L OW
W ER L D
DH IH S IH Z AH T EH S T
HH AY TH EH R
HH AW AA R Y UW
AY L AH V P AY TH AH N
N AE CH ER AH L L AE NG G W AH JH
P R OW S EH S IH NG
IH Z F AH N
M AH SH IH N L ER N IH NG
IH Z P AW ER F AH L
K AA M P Y UW T ER S AY AH N S
IH Z IH N T ER EH S T IH NG
W IY K AE N D UW IH T
L EH T S G OW
S T AA R T IH NG N AW

EXAMPLE:
the count of 'HH' is 3, which is the same in llm model.
the count of 'HH EH' is 1, which is the same in llm model.
the count of 'HH AY TH' is 1, which is the same in llm model.
Overall, we can find that llm model counts n-grams correctly.
```

### Does your code count n-grams in the same way?

```
Answer here.
No.
If the answer is no, explain why your answer is better or worse.
My code is worse in tiny training set since we set the unk threshold to be 20, then all the tokens in tiny_training.txt will be replaced by <unk>. However, My code is better in large training set since the unk threshold is 20 rather than 1 in chatgpt code. Therefore, it can reduce more data sparsity.  
```


----


## Is the model produced by the LLM correct?

```
Answer here.
Explain (with examples if needed) why or why not.
HINT: consider both the format and the content as specified in assignment 3.
The mode produced by the LLM is like:
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
Therefore, we can see that all the n-gram models are included in the json file. Additionally, we know LLM counts n-grams correctly so that the model produced by the LLM correct.
```

### Is the model produced by your code different?

```
Answer here.
yes
If the answer is no, explain why.
If the answer is yes, explain why your answer is better or worse.
```
My answer is worse than chatgpt code since my code only provides the information of unigram, bigram and trigram models. However, the chatgpt code provides additional information of "meta" containing bos, eos, unk, utterances, unigram total count and vocabulary size.
----


# Other LLM code that is **incorrect**

```
Answer here. Explain with examples.
N/A
Write N/A if not applicable.
```
