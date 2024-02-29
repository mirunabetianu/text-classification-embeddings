# Text Classification with Embeddings

## Introduction

This is an interesting problem that goes in the area of extreme classification, given the high number of classses that both train and test datasets have. In the train dataset, there are **1565 classes**, whereas in the test dataset **1538 classes**. Moreover, a large number of classes from the test set are **not** present in the training set. Thus, the solution of this problem might not be the training of a simple multi-label classifier (or maybe it could be using heavy clustering techniques), but rather finding out how we can leverage the (sentence/word) embeddings to find out one correct category for each product. 

## Requirements
- Python 3.8+
- NumPy
- sklearn
- Install with `pip install -r requirements.txt`

## Implementation Steps

### Step 1: Data Loading
- `load_json_file(path)`: Loads and returns data from a JSON file.
- `load_glove_embeddings(path)`: Loads Glove embeddings from a file.

### Step 2: Preprocessing
- `preprocess_sentence(sentence)`: Removes punctuation and converts the sentence to lowercase.
  

### Step 3: Sentence Embeddings
- `compute_sentence_embeddings(sentence_batch, word_embeddings)`: Computes sentence embeddings for a batch of sentences by averaging the word embeddings. Words not in the Glove Embedding collection are omitted.
- Tried weighted average by computing the **tfidf** scores of the product names; not included in the final solution since it had a negative impact on the accuracy. 

### Step 4: Category Embeddings
- `compute_category_embeddings(products, word_embeddings, batch_size=batch_size)`: Computes category embeddings as an average of the sentences embeddings from that category.


### Step 5: Matching Categories
-  `match_categories(product_embeddings, category_embeddings, n_matches=6)`: Computes the cosine similarity between product and category embeddings, returning the top N matches.

### Step 7: Finding Out Accuracy
- `compute_accuracy(predictions, test_products)`: Computes accuracy on the test set. As mentioned in the task requirements, **the classification is successful when the solution predicts (at least) one correct class.**

## Results

The accuracy is heavily influenced by the number of classes we want to predict for a products. In this case, I selected the first 6 matches, since it the maximum number of categories that a product could have in the training set. By selecting the first 6 matches, the accuracy is 83.39%:

 <img src="https://i.ibb.co/c33pRYZ/Screenshot-2024-02-28-at-16-12-43.png" alt="83.39%" />

If we want to predict a singular class per product, the accuracy goes down to 56.80%, which is still fair given the large number of classes.
