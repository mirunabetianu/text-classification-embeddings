# Text Classification with Embeddings

## Introduction

In this repo you can find my implementation of the Jr Data Engineer Challenge. The goal is to leverage word embeddings to categorize unseen text data into predefined classes.

Even though at first seemed like an easy task, once you start to tackle it, you realise it's not your average classification problem. The extremely high number of classes and their sparsity across the samples were two of the main challenges of this project.

My implementation is organized as follows:

#### python-cosine-similarity
- Implementation of the task using cosine similarity to match the category and sentence embeddings in `python`.
#### python-lstm
- Implementation of the task using a simple LSTM in `python`.
#### rust-cosine-similarity-wip
- Implementation of a small part of the task in `rust`. This section is a WIP since I was really curious of how to implement this in `rust`, but given time constraints I couldn't really finish. Here, I computed the sentence embeddings in `rust`! :)
  

For every version you can find a small description in each folder. The LSTM version does not follow the exact steps from the task description, thus I was not sure if it can be valid. 

(!) In every folder it is assumed that the files `products.json`, `products.test.json` and `glove.6B.100d.txt` are present.

## Future improvements 
The code can definitely benefit from some improvements, even if they are code or methodology related. Out of the top of my mind, some of them would be:
- Adding unit tests to functions to ensure edge cases in the data are handled (i.e., the product name is _None_)
- Improving the computation of sentence embeddings
  - Since the tfidf scores as weights for averaging did not add any improvement, (maybe) we can compute the average of the most _n_ relevant words in the sentence instead.
  - Checking out other possible weights for the word embeddings
  - Not discarding the words that are not present in the Glove Embedding and computing their embedding in a way (i.e., subword embedding)
- Checking out clustering methods
  -  Grouping the most connected category embeddings in clusters, hence reducing the dimensionality of the prediction task
- Experimenting with LLMs (my poor CPU was already giving up with the LSTM)

Thank you for taking the time reading this and I look forward to discussing this assignment with you on Monday!