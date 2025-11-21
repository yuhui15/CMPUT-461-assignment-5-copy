
# Answer these questions with respect to the data in the `dev` set.

## Does the LLM handle begin and end of sentence tokens across models correctly?

```
Answer here.
No. The LLM doesn't handle begin and end of sentence tokens across models correctly.
Explain (with examples if needed) why or why not.
The code only turns the utterance into [<s>] + utterance + [</s>] for unigram, bigram, trigram models.
The correct method is to turn the utterance into utterance, [<s>] + utterance + [</s>], and [<s>, <s>] + utterance + [</s>, </s>] for unigram, bigram and trigram models respectively. If we use LLM code, we caonnot compute the probability of the first token of the utterance with trigram model. 
```

### Does your code handle these tokens in the same way?

```
Answer here.
No.
If the answer is no, explain why your answer is better or worse.
My answer is better since my method is to turn the utterance into utterance, [<s>] + utterance + [</s>], and [<s>, <s>] + utterance + [</s>, </s>] for unigram, bigram and trigram models respectively. Therefore, we can compute the probability of the first token of the utterance with trigram model.
```

----


## Does the LLM count n-grams correctly?

```
Answer here.
Explain (with examples if needed) why or why not.
HINT: use the tiny models.
I have manually check the tiny models trained on tiny_training.txt with tiny_dev.txt
The following is the content of tiny_dev.txt:
W AH T AH G R EY T D EY
DH AH S AH N IH Z SH AY N IH NG
B ER D Z AA R S IH NG IH NG
K AE T S M IY AW

EXAMPLE:
the count of 'W' is 3, which is the same in llm model.
the count of 'W AH' is 1, which is the same in llm model.
the count of 'W AH T' is 0, which is the same in llm model.
Overall, we can find that llm model counts n-grams correctly.
```

### Does your code count n-grams in the same way?

```
Answer here.
If the answer is no, explain why your answer is better or worse.
My code is worse in tiny training set since we set the unk threshold to be 20, then all the tokens in tiny_training.txt will be replaced by <unk>. However, My code is better in large training set since the unk threshold is 20 rather than 1 in chatgpt code. Therefore, it can reduce more data sparsity.  
```


----

## Does the LLM handle OOV tokens correctly?

```
Answer here.
Explain (with examples if needed) why or why not.
The chatgpt code replaces all the unseen tokens in dev set with <unk> label. Since we already have <unk> label in vocabulary set, it can assume that the unseen tokens is equivalent to <unk> which is in vocabulary set. Therefore, there is no OOV tokens.
```

### Does your code handle OOV tokens in the same way?

```
Answer here.
Almost the same.
If the answer is no, explain why your answer is better or worse.
My code is worse in tiny training set since we set the unk threshold to be 20, then all the tokens in tiny_training.txt will be replaced by <unk>. However, My code is better in large training set since the unk threshold is 20 rather than 1 in chatgpt code. Therefore, it can reduce more data sparsity.  
```


----

## Does the LLM compute perplexity correctly?

```
Answer here.
No.
Explain (with examples if needed) why or why not.
It calculates the non-zero probability of each token depending on the given n-gram method (unigram, bigram, trigram) and smooth method (laplace) correctly. However, in chatgpt code, when computing perplexity, it will set the perplexity score to be 'inf' when the probability of the n-gram is 0. Therefore, it will cause the perplexity to be incorrect.
```

### Does your code compute perplexity in the same way?

```
Answer here.
No.
If the answer is no, explain why your answer is better or worse.
It calculates the non-zero probability of each token depending on the given n-gram method (unigram, bigram, trigram) and smooth method (laplace) correctly. In my code, when computing perplexity, it will use Back-off algorithms when the probability of the n-gram is 0 to set it to 1/|V| where |V| is the vocabulary size. Therefore, my code is better than chatgpt code.
```


----

## Does the LLM perform add-one smoothing corectly?

```
Answer here.
Yes.
Explain (with examples if needed) why or why not.
Chatgpt code uses
'''
    if laplace:
        return (num + 1) / (den + V)
'''
to apply add-one somoothing and we can see that the formula is correct.
```

### Does your code perform add-one smoothing in the same way?

```
Answer here.
Yes
If the answer is no, explain why your answer is better or worse.
```


----

# Other LLM code that is **incorrect**

```
Answer here. Explain with examples.
N/A
Write N/A if not applicable.
```