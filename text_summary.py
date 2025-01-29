# Import necessary libraries
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from heapq import nlargest
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import spacy

# Load spaCy for sentence tokenization
nlp = spacy.load('en_core_web_sm')

# Load a pre-trained Transformer model for sentence embeddings
# Here, we use the 'distilbert-base-uncased' model for efficiency
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Initialize the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_sentence_embeddings(sentences):
    """
    Generate sentence embeddings using DistilBERT.
    """
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token embedding as the sentence embedding
    sentence_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return sentence_embeddings

def summarizer(rawdocs, summary_percent=0.6):
    """
    Advanced summarizer using Transformer-based sentence embeddings.
    """
    # Tokenize the input text into sentences
    doc = nlp(rawdocs)
    sentences = [sent.text for sent in doc.sents]

    # Get sentence embeddings
    sentence_embeddings = get_sentence_embeddings(sentences)

    # Compute the document embedding (average of all sentence embeddings)
    doc_embedding = np.mean(sentence_embeddings, axis=0).reshape(1, -1)

    # Calculate sentence scores (cosine similarity with the document embedding)
    sentence_scores = []
    for sent_embedding in sentence_embeddings:
        score = cosine_similarity(sent_embedding.reshape(1, -1), doc_embedding)[0][0]
        sentence_scores.append(score)

    # Boost scores for sentences containing named entities
    important_entities = set(ent.text for ent in doc.ents)
    for i, sent in enumerate(doc.sents):
        for word in sent:
            if word.text in important_entities:
                sentence_scores[i] += 1  # Boost score

    # Select top N sentences
    select_length = max(1, int(len(sentences) * summary_percent))
    top_sentence_indices = nlargest(select_length, range(len(sentence_scores)), key=lambda i: sentence_scores[i])

    # Sort selected sentences back to their original order
    top_sentence_indices.sort()

    # Generate the final summary
    final_summary = ' '.join([sentences[i] for i in top_sentence_indices])

    # Return the summary, original document, and lengths
    return final_summary, doc, len(rawdocs.split()), len(final_summary.split())