
## Assignment 3 URL

Complete your assignment 3 URL:

https://github.com/CMPUT-461-561/f25-asn3-yuhui15


----

## Choice of LLM to compare against:

```
ChatGPT
```

### Brief comments

```
Aspect #1:
In my code, there are two functions: train_model(train_file, output_file) and evaluate_model(method, vocab_file, test_file, smooth).
The train_model function train the unigram, bigram and trigram models based on the train_file and they are written to the output_file.
The evaluate model function evaluate the model from vocab_file with test_file depending on the arguments (unigram, bigram, trigram) and (--laplace, none). 
In chatgpt code, there are six functions ( read_utterances(path), build_vocab(utterances), replace_singletons_with_unk(utterances, vocab), count_ngrams(utterances), convert_to_json(counter), create_model_json(uni, bi, tri, vocab, n_utterances) in train.py which are equivalent to train_model function in my code.
In chatgpt code, there are seven functions ( read_utterances(path), map_oov(tokens, vocab), load_model(path), p_unigram(w, model), p_bigram(w1, w2, model, laplace), p_trigram(w1, w2, w3, model, laplace), compute_ppl(model_type, model, utterances, laplace, vocab) ) in eval.py which are equivalent to evaluate_model function in my code.
Moreover, we can see that chatgpt source code has train.py, eval.py, and my source code also has train.py, eval.py. However, the claude has one more py script utils.py which is a helper script for the three models. The chatgpt code and my code both have the part to process the oov words with <unk>, while claude code doesn't have the part. 
 
Aspect #2: 

Both my code and chatgpt code both minimize the use of the type annotations and keep code style to be clear and easy to read. 
Both my code and chatgpt code have variable names which can help readers understand how each function works. (e.g., (unigram, count_ngrams, compute_ppl) in chatgpt code, (unigrams_dict, bigrams_dict, trigrams_dict) in my code.
Both my code and chatgpt code doesn't have redundant content so that it can help debug.
```


----


## "Copy-and-paste" execution instructions to run the **LLM code** after your changes

```
The command line to use my code to train the model using tiny_training.txt
% python3 src/mycode/train.py data/tiny_training.txt tiny_model_mine.json

The command line to use llm code to train the model using tiny_training.txt
% python3 src/chatgpt/train.py data/tiny_training.txt tiny_model_llm.json

The command line to use llm code to train the model based on training.txt
% python3 src/chatgpt/train.py data/training.txt data/training.json

The command line to use llm code to evaluate the model trained on training.txt
% python3 src/chatgpt/eval.py unigram data/training.json data/training.txt
% python3 src/chatgpt/eval.py bigram  data/training.json data/training.txt
% python3 src/chatgpt/eval.py bigram  data/training.json data/training.txt --laplace
% python3 src/chatgpt/eval.py trigram data/training.json data/training.txt
% python3 src/chatgpt/eval.py trigram data/training.json data/training.txt --laplace

% python3 src/chatgpt/eval.py unigram data/training.json data/dev.txt
% python3 src/chatgpt/eval.py bigram  data/training.json data/dev.txt
% python3 src/chatgpt/eval.py bigram  data/training.json data/dev.txt --laplace
% python3 src/chatgpt/eval.py trigram data/training.json data/dev.txt
% python3 src/chatgpt/eval.py trigram data/training.json data/dev.txt --laplace
```


----

## YOUR Evaluation **copied** from what you submitted in assignment 3

|Model           | Smoothing  | Training set PPL | Dev set PPL |
|----------------|----------- | ---------------- | ----------- |
|unigram         | -          |      38.06       |    38.06    |
|bigram          | unsmoothed |      18.61       |    18.60    |
|bigram          | Laplace    |      36.42       |    36.63    |
|trigram         | unsmoothed |      9.36        |    9.46     |
|trigram         | Laplace    |      14.92       |    15.15    |



## LLM evaluation

|Model           | Smoothing  | Training set PPL | Dev set PPL |
|----------------|----------- | ---------------- | ----------- |
|unigram         | -          |      35.38       |    35.25    |
|bigram          | unsmoothed |      15.55       |    inf      |
|bigram          | Laplace    |      16.23       |    16.21    |
|trigram         | unsmoothed |       7.00       |    inf      |
|trigram         | Laplace    |      10.78       |    10.90    |

<img width="1195" height="402" alt="image" src="https://github.com/user-attachments/assets/60313f06-d729-4a58-9194-8917c25785f8" />


