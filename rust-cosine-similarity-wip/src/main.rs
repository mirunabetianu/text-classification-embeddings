use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Deserialize)]
struct Product {
    name: Option<String>,
    category: Vec<String>,
}

fn main() {
    let products = read_products_from_file("products.json");
    let _test_products = read_products_from_file("products.test.json");
    let word_embeddings = read_word_embeddings_from_file("glove.6B.100d.txt");

    let product_names: Vec<_> = products.iter().map(|p| p.name.clone()).collect();
    for name in product_names.iter() {
        println!("{:?}", name);
        let product_name_embedding = compute_sentence_embeddings(name.clone(), &word_embeddings);
        println!("{:?}", product_name_embedding);
    }
}

fn read_products_from_file(file_path: &str) -> Vec<Product> {
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    serde_json::from_reader(reader).unwrap()
}

fn read_word_embeddings_from_file(file_path: &str) -> HashMap<String, Vec<f32>> {
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    let mut word_embeddings = HashMap::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let mut parts = line.split_whitespace();
        let word = parts.next().unwrap().to_string();
        let embedding: Vec<f32> = parts.map(|p| p.parse().unwrap()).collect();
        word_embeddings.insert(word, embedding);
    }
    word_embeddings
}

fn preprocess_sentence(sentence: &str) -> Vec<String> {
    let re = Regex::new(r"\b\w+\b").unwrap();
    re.find_iter(sentence)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

fn compute_sentence_embeddings(
    sentence: Option<String>,
    word_embeddings: &HashMap<String, Vec<f32>>,
) -> Vec<f32> {
    if let Some(sentence) = sentence {
        let tokens = preprocess_sentence(&sentence);
        let tokens_embeddings: Vec<_> = tokens
            .iter()
            .filter_map(|token| word_embeddings.get(token))
            .collect();
        let sentence_embedding = if tokens_embeddings.is_empty() {
            vec![0.0; 100]
        } else {
            let sum: Vec<f32> = tokens_embeddings.iter().fold(vec![0.0; 100], |acc, x| {
                acc.iter().zip(x.iter()).map(|(a, b)| a + b).collect()
            });
            sum.iter()
                .map(|x| x / tokens_embeddings.len() as f32)
                .collect()
        };
        return sentence_embedding;
    }
    vec![0.0; 100]
}
