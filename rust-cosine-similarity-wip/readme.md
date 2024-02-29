# Text Classification with Embeddings

## Introduction

The transition from `python` to `rust` was quite the cultural shock. Initially I thought that it couldn’t be that hard, but was I in for a surprise. `rust` welcomed me with open arms and humbled me in the next second. However, it was really fun to work with it and here’s my work-in-process implementation so far.

## Implementation Steps

### Step 1: Data Loading
- `read_products_from_file(file_path: &str) -> Vec<Product> `: Loads and returns data from a JSON file.
- `read_word_embeddings_from_file(file_path: &str) -> HashMap<String, Vec<f32>>`: Loads Glove embeddings from a file.

### Step 2: Preprocessing
- `preprocess_sentence(sentence: &str) -> Vec<String> `: Removes punctuation and converts the sentence to lowercase.
  

### Step 3: Sentence Embeddings
- `compute_sentence_embeddings(
    sentence: Option<String>,
    word_embeddings: &HashMap<String, Vec<f32>>,
) -> Vec<f32>`: Computes sentence embeddings for a sentence by averaging the word embeddings. Words not in the Glove Embedding collection are omitted.

