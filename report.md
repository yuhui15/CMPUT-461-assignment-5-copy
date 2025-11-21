# Report
The goal of this assignment is for me to compare and contrast code written by me and code written by an LLM.
All conclusions are based on my answers to architecture.md, style.md, most_similar.md, most_dissimilar.md, training.md, and evaluation.md, the JSON files created by my code and by the LLM and they refer to the perplexity scores computed by the LLM. 

## Correctness
The LLM doesn't write code with fewer errors than me. The evidence from training.md and evaluation.md, architecture.md and perplexity table can support my conclusion. 

(1) In training.md (training stage), we can find that the chatgpt code only turns the utterance into ```[<s>] + utterance + [</s>]``` for unigram, bigram, trigram models. If we use LLM code, we cannot compute the count of the first token of the utterance for trigram model. However, my code is better since my method is to turn the utterance into utterance, ```[<s>] + utterance + [</s>]```, and ```[<s>, <s>] + utterance + [</s>, </s>]``` for unigram, bigram and trigram models respectively. Therefore, we can compute the count of the first token of the utterance for trigram model on training set.

(2) In evaluation.md (dev stage), we can find that the chatgpt code only turns the utterance into ```[<s>] + utterance + [</s>]``` for unigram, bigram, trigram models. If we use LLM code, we cannot compute the probability of the first token of the utterance with trigram model. However, my code is better since my method is to turn the utterance into utterance, ```[<s>] + utterance + [</s>]```, and ```[<s>, <s>] + utterance + [</s>, </s>]``` for unigram, bigram and trigram models respectively. Therefore, we can compute the probability of the first token of the utterance with trigram model.

(3) In evaluation.md (dev stage), we can find that the chatgpt code calculates the non-zero probability of each token depending on the given n-gram method (unigram, bigram, trigram) and smooth method (laplace) correctly. However, in chatgpt code, when computing perplexity, it will set the perplexity score to be 'inf' when the probability of the n-gram is 0. Therefore, it will cause the perplexity to be incorrect. In the perplexity score table computed by LLM, we can see "bigram unsmoothed" and "trigram	unsmoothed" both have the results 'inf' which is not correct. Comparing with my code, when computing perplexity, it will use Back-off algorithms when the probability of the n-gram is 0 to set it to 1/|V| where |V| is the vocabulary size. Therefore, my code is better than chatgpt code. In the perplexity score table computed by my code, we can see all the computing methods have the correct results rather than 'inf'.


## Style
My code's style is better than LLM's. The evidence from style.md, architecture.md can support my conclusion. 

(1) In style.md, we can find that my code's comments provide more details of pipeline for each function, while chatgpt code only use one sentence or phrase to summarize the content of the function. My code also provides the comment to summarize each python script train.py, eval.py, while chatgpt code doesn't.

(2) In style.md, we can find that my code prints "Processing utterances" and Model Statistics" status in training stage while chatgpt code doesn't. Moreover, my code prints " Initalization of Data", "Processing Files & Transforming to Phonemes", "Splitting Data", " Preparing Files", status in evaluating stage while chatgpt doesn't. 

(3) In style.md, we can find that my code add "Perplexity score" in front of the perplexity score which will let readers know what the number represents.

## Similarity
The random people can tell my code is not written by LLM. The evidence from most_similar.md and most_dissmilar.md, architecture.md can support my conclusion. 

(1) In most_similar.md, we can see that in my code, there is only one function "train_model" to train the model.However, there are several functions containing "read_utterances", build_vocab, replace_singletons_with_unk, count_ngrams, convert_to_json, create_model_json to train the model. Therefore, we find that the size of each module in my code is larger than one in chatgpt code.

(2) In most_dissmilar.md, in my code, there is only one function "evaluate_model" to evaluate the model.
However, there are several functions containing "read_utterances", "map_oov", "load_model", "p_unigram", "p_bigram", "p_trigram", "compute_ppl" to evaluate the model. Therefore, we find that the size of each module in my code is larger than one in chatgpt code.

## Should computing assignments like assignment 3 be given for marks?
The assginment 3 can still be given for marks since LLM can have the error in generating the code. Therefore, the students cannot be completely dependent on LLM without checking the error. The evidence from training.md and report.md can support my conclusion.

(1) Based on the evidence from Correctness part, we can find that chatgpt code has errors in handling begin and end of sentence tokens across models, computing perplexity ('inf' may be the results of evaluating). This means LLM cannot finish the task without any human's debugging. 

(2) Based on the evidence from Style part, we can find that chatgpt code doesn't print additional information of comments, print statements and the perplexity results for human to have a better understanding.

(3) Based on the evidence from similarity,  we can find that the size of each module in my code is larger than one in chatgpt code.
