from collections import defaultdict
import numpy as np 
import json
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import MultiLabelBinarizer
from ltsm_model import LSTMModel

embedding_size = 100
batch_size = 32
hidden_dim = 128  
num_layers = 2  
num_epochs = 3

def load_json_file(path):
    """Loads content of JSON file

    Args:
        path (string): The path of the JSON file

    Returns:
        dict: Dictionary containing the JSON data
    """  
    with open(path, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)
    return data

def load_glove_embeddings(path):
    """Loads content of Glove Embedding file

    Args:
        path (string): The path of the Glove file

    Returns:
        dict: Dictionary having a word as key and its corresponding embedding as value
    """
    word_embeddings = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            values = line.split()
            word = values[0]
            word_embedding = np.asarray(values[1:], dtype=np.float32)
            word_embeddings[word] = word_embedding
    return word_embeddings

def preprocess_sentence(sentence):
    """Processing a sentence to remove any non-alphabetical character

    Args:
        sentence (string): The name of a product

    Returns:
        list of strings: list of words/tokens 
    """
    words = sentence.split()
    words_without_punctuation = [re.sub(r'[^a-zA-Z]', '', word).lower() for word in words]
    return [word for word in words_without_punctuation if word]

def sentence_to_embedding_sequence(sentence, embeddings_dict, max_len=None):
    """Compute a sequence of embeddings for a sentence (product name)

    Args:
        sentence (string): name of a product
        embeddings_dict (dict): dict containing the Glove Embeddings
        max_len (int, optional): number indicating the maximum length of the embedding sequence. 

    Returns:
        tensor: the sequence of embeddings corresponding to the sentence
    """
    if sentence is not None:
        words = preprocess_sentence(sentence)
        embedding_sequence = [torch.tensor(embeddings_dict[word]) for word in words if word in embeddings_dict]
        
        # Having an uniform size for the embedding sequences is crucial for the LTSM
        if max_len is not None:
            if len(embedding_sequence) > max_len:
                embedding_sequence = embedding_sequence[:max_len]
            elif len(embedding_sequence) < max_len:
                # Padding with 0s in case the sentence is shorter than the max
                padding = [torch.zeros(embedding_size) for _ in range(max_len - len(embedding_sequence))]  
                embedding_sequence += padding
        
        return torch.stack(embedding_sequence)
    return torch.zeros(embedding_size)

# Loading train and test datasets
train_products = load_json_file('products.json')
test_products = load_json_file('products.test.json')

# Loading the Glove Embeddings
word_embeddings = load_glove_embeddings('glove.6B.100d.txt')

# Used a Binarizer for generating the one-hot-labels 
# A nicer version compared to the manual one, helps when trying to create a binary column for every class given as a list of classes
mlb = MultiLabelBinarizer()
train_one_hot_labels = mlb.fit_transform([item['category'] for item in train_products])

num_classes = len(mlb.classes_)
# Computing the length of the longest sentence
max_sequence_length = max([len(item['name'].split()) for item in train_products if item['name'] is not None])

# Computing the sentence embedding sequence
# Since the order of the words MAY be important, I preserved the order
train_embedding_sequences = [sentence_to_embedding_sequence(item['name'], word_embeddings, max_sequence_length) for item in train_products]
# Additional padding to make the sequences of the same size is necessary for LTSM
padded_sequences = pad_sequence(train_embedding_sequences, batch_first=True)
# Creating standard TensorDataset and DataLoader
train_labels_tensor = torch.tensor(train_one_hot_labels, dtype=torch.float32)
train_dataset = TensorDataset(padded_sequences, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Initializing the model, using BCE for multi-label and Adam (default) as optimizer 
model = LSTMModel(embedding_size, num_classes, hidden_dim, num_layers)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    print(epoch)
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

# Evaluation function to compute the accuracy of the LTSM
# Since the model outputs logits, I applied sigmoid for a better fit between 0 and 1 of the predicted probabilities
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    correct_preds = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            predicted = outputs.sigmoid() > 0.5
            correct_preds += (predicted == labels.float()).sum()
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_preds / (len(data_loader.dataset) * num_classes)
    return avg_loss, accuracy

# Test set has the same processing steps as the train set
# Here, I use the function !transform! of the binarizer to ensure that the classes are encoded the same 
# The extra classes in the test set are ignored
test_one_hot_labels = mlb.transform([item['category'] for item in test_products])
test_embedding_sequences = [sentence_to_embedding_sequence(item['name'], word_embeddings, max_sequence_length) for item in test_products]
padded_sequences = pad_sequence(test_embedding_sequences, batch_first=True)
test_labels_tensor = torch.tensor(test_one_hot_labels, dtype=torch.float32)
test_dataset = TensorDataset(padded_sequences, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Finding out accuracy 
test_loss, test_accuracy = evaluate(model, test_loader)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")



