# Text Classification with Embeddings

## Introduction

Even though the previous solution performs relatively well, when dealing with tricky text data, the performance may be improved by using a NN. Since the product names are represented by rather shorter strings with a specific structure "Name - Short description - Color", I thought that the order of which the words appear in the sentence may have a large impact on the performance. So, instead of computing the sentence embeddings, I computed sentence embedding sequences and passed them through a LTSM.  The only _issue_ with approach is that it deviates from the steps provided in the task, therefore I added it as an option. 

## Requirements
- Python 3.8+
- NumPy
- sklearn
- pytorch
- Install with `pip install -r requirements.txt`

## Implementation Steps

### Step 1: Data Loading
- `load_json_file(path)`: Loads and returns data from a JSON file.
- `load_glove_embeddings(path)`: Loads Glove embeddings from a file.

### Step 2: Preprocessing
- `preprocess_sentence(sentence)`: Removes punctuation and converts the sentence to lowercase.
  

### Step 3: Sentence Embeddings Sequences
- `sentence_to_embedding_sequence(sentence, embeddings_dict, max_len=None)`: Computes sentence embedding sequence for a given sentence, returns a list with word embeddings as they appear in the sentence along with padding if necessary 


### Step 4: Train the LTSM 
-  Prepare train dataset for the LTSM (i.e., converting to tensors, addition of padding)
-  Define the model, hyperparameters, loss and optimizer 


### Step 5: Evaluate the LTSM
-  Prepare test dataset for the LTSM (same type of processing as for the training)
-  Get predictions

### Step 7: Finding Out Accuracy
- Compare the predicted classes with the true classes from the dataset 

## Results

The accuracy on the test dataset is 99.7%.

 <img src="https://i.ibb.co/vXTxbB1/Screenshot-2024-02-28-at-16-11-53.png" alt="99.7%" />

