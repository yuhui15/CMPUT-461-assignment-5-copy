# Discuss similarities and differences in coding style

----

### Comments

**Example:**

|yours|LLM|
|---|---|
|<img width="662" height="442" alt="image" src="https://github.com/user-attachments/assets/24a647f9-e4cd-42c6-8b45-0b8407b26dd8" />|<img width="547" height="317" alt="image" src="https://github.com/user-attachments/assets/3d561db6-ec7a-4179-969d-f6364feb9643" />|

**Discussion:**

```
Write your answer here.
Similarities: My code and chatgpt code both have the comments to describe each function.
Differences: My code's comments provide more details of pipeline for each function, while chatgpt code only use one sentence to summarize the content of the function.
             My code also provides the comment to summarize each python script train.py, eval.py, while chatgpt code doesn't.
```

----

### Function declarations

**Example:**

|yours|LLM|
|---|---|
|<img width="387" height="26" alt="image" src="https://github.com/user-attachments/assets/8a6dd6a4-788b-48e0-8c02-a5f2e29dc2a0" />|<img width="297" height="32" alt="image" src="https://github.com/user-attachments/assets/6edeb065-943b-4254-96ef-78b7c88ffe84" />|
|<img width="553" height="20" alt="image" src="https://github.com/user-attachments/assets/2ae6e8f1-9cfd-4fe7-abac-2e40e5c7ad0a" />|<img width="458" height="30" alt="image" src="https://github.com/user-attachments/assets/78e27412-25e1-4081-a726-09b89234ce1c" />|


**Discussion:**

```
Write your answer here.
Similarities: Function declarations in my code and chatgpt code are both easy for human to understand each function's content.
Differences: We only have two function declarations in my code which have the high-level summary of my code. However, the chatgpt code has multiple function declarations
             which divide the entire task into several small sub-tasks and make it easier to read the function declarations one by one to know the steps.
```

----

### Organization of the JSON file

**Example:**

|yours|LLM|
|---|---|
|<img width="425" height="147" alt="image" src="https://github.com/user-attachments/assets/9d8461e7-473b-4403-9ac3-dc9cc4debd5c" />|<img width="552" height="547" alt="image" src="https://github.com/user-attachments/assets/1a7a61f2-6fbf-4dfa-8f63-aaf51e2f4729" />|

**Discussion:**

```
Write your answer here.
Similarities: My code and chatgpt code both contain unigram, bigram and trigram model in JSON file.
Differences: The JSON file have additional information "meta" which contain bos, eos, unk, utterances, unigram total count and vocabulary size. 
```

----

### Print statements

**Example:**

|yours|LLM|
|---|---|
|<img width="625" height="311" alt="image" src="https://github.com/user-attachments/assets/3e08ce10-1034-45d5-a03b-4c565103bb56" />|<img width="446" height="45" alt="image" src="https://github.com/user-attachments/assets/238f28bd-dc6c-4228-8393-7b733ee62893" />|

**Discussion:**

```
Write your answer here.
Similarities: My code and chatgpt code both print status of finishing creating the model and the perplexity score.
Differences:
(1) My code prints "Processing utterances" and Model Statistics" status in training stage while chatgpt code doesn't.
(2) My code prints " Initalization of Data", "Processing Files & Transforming to Phonemes", "Splitting Data", " Preparing Files", status in evaluating stage while chatgpt doesn't.
```

----

### Presenting the perplexity results

**Example:**

|yours|LLM|
|---|---|
|<img width="487" height="33" alt="image" src="https://github.com/user-attachments/assets/f66b57bb-63cc-4ce2-a18e-b965bfc20934" />|<img width="128" height="31" alt="image" src="https://github.com/user-attachments/assets/0c547733-f2a7-4491-b4aa-ce4e285393a5" />|

**Discussion:**

```
Write your answer here.
Similarities: My code and chatgpt code both print perplexity results after computing the score.
Differences: My code add "Perplexity score" in front of the perplexity score.
```
