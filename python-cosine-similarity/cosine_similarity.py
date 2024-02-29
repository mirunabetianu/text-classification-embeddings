import json
import re
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

embedding_size = 100 
batch_size = 512
n_matches = 6 # computed previously as the max number of categories a product can have in the train set

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

def compute_sentence_embeddings(sentence_batch, word_embeddings):
    """Computing sentence embedding as an average of its containing words' embeddings

    Args:
        sentence_batch (list of string): list containing multiple product names
        word_embeddings (dict): dict containing the Glove Embeddings

    Returns:
        ndarray: np array containing the embeddings of each of the processed sentences
    """
    sentence_batch_embeddings = []

    for sentence in sentence_batch:
        if sentence is not None:
            tokens = preprocess_sentence(sentence)
            tokens_embeddings = [word_embeddings[token] for token in tokens if token in word_embeddings]
            if tokens_embeddings:
                sentence_embedding = np.mean(tokens_embeddings, axis=0)
            else:
                # Placeholder embedding for sentences containing words not found in Glove
                sentence_embedding = np.zeros(embedding_size)
        else:
            # Placeholder embedding for empty sentences (some are None)
            sentence_embedding = np.zeros(embedding_size)
        sentence_batch_embeddings.append(sentence_embedding)

    return np.array(sentence_batch_embeddings)

def compute_category_embeddings(products, word_embeddings, batch_size=batch_size):
    """Computing category embedding as an average of their corresponding sentences embeddings

    Args:
        products (dict): dict containing the train product names and their categories
        word_embeddings (dict): dict containing the Glove Embeddings
        batch_size (int, optional): number of sentences to process collectively. 

    Returns:
        dict: dict containing the category as key and their corresponding embedding as value
    """
    category_embeddings = defaultdict(list)

    for i in range(0, len(products), batch_size):
        # Using a batch sentence embedding computation for faster runtime
        batch = products[i:i+batch_size]
        batch_product_names = [product['name'] for product in batch]
        batch_embeddings = compute_sentence_embeddings(batch_product_names, word_embeddings)

        for index, product in enumerate(batch):
            category_embedding = batch_embeddings[index]
            for category in product['category']:
                category_embeddings[category].append(category_embedding)

    for category, category_embedding in category_embeddings.items():
        category_embeddings[category] = np.mean(category_embedding, axis=0)

    return category_embeddings


def match_categories(product_embeddings, category_embeddings, n_matches=n_matches):
    """Compute best similarity score between product embeddings and category embeddings

    Args:
        product_embeddings (ndarray): np array containing the embeddings of the product names
        category_embeddings (dict): dict containing the Glove Embeddings
        n_matches (int, optional): number of best matches to return. Defaults to n_matches.

    Returns:
        list: list containing the best n_matches for every product
    """
    categories = list(category_embeddings.keys())
    categories_embeddings_matrix = np.array([category_embeddings[category] for category in categories])

    similarity_scores = cosine_similarity(product_embeddings, categories_embeddings_matrix)

    best_similarity_matches_indices = np.argsort(similarity_scores, axis=1)[:, -n_matches:]

    best_similarity_matches = []

    for indices in best_similarity_matches_indices:
        current_product_categories = []
        
        for index in reversed(indices):
            current_product_categories.append(categories[index])
        
        best_similarity_matches.append(current_product_categories)
    
    return best_similarity_matches

def compute_accuracy(predictions, test_products):
    """Compute accuracy based on having one correct predicted class for a product

    Args:
        predictions (list): list containing an array of n_matches for every product
        test_products (dict): dict containing the test product names and their categories

    Returns:
        int: accuracy score 
    """
    correct_predictions = 0

    for index, prediction in enumerate(predictions):
        true_categories = test_products[index]['category']

        correct = any(predicted_category in true_categories for predicted_category in prediction)
        if correct:
            correct_predictions +=1

    accuracy = correct_predictions/len(test_products)
    return accuracy 


# Loading train and test datasets
train_products = load_json_file('products.json')
test_products = load_json_file('products.test.json')

# Loading Glove Embeddings
word_embeddings = load_glove_embeddings('glove.6B.100d.txt')

# Computing category embeddings 
category_embeddings = compute_category_embeddings(train_products, word_embeddings)

# Computing the embeddings of the test sentences
test_products_names = [item['name'] for item in test_products if item['name']]
test_products_embeddings = compute_sentence_embeddings(test_products_names, word_embeddings)

# Find best similarity matches between category and product embeddings
predictions = match_categories(test_products_embeddings, category_embeddings)

# Find out accuracy 
accuracy = compute_accuracy(predictions, test_products)

print(f"Overall Accuracy: {accuracy * 100:.2f}%")
