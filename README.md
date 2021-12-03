# Emotion-Cause-Extraction

### Some of the Required Libraries
pandas

numpy

sklearn

benepar 

pycorenlp

### Download en_core_web_trf
python -m spacy download en_core_web_trf

### Source Files 
clause_extraction_and_data_annotation.py - Used to extract clauses and create the annotated dataset.<br>
sentimentAwareWE.ipynb - Used to create emotion-aware embeddings.<br>
model.py - Trains the emotion extraction and cause extraction model on the annotated dataset and creates the predicted file.<br>
cluster_review_clauses.py - Takes the dataset with predicted cause clauses and outputs the selected cause clauses for each (product, emotion) pair

### Sentiment/Emotion Aware Word Embeddings

Using initial Word2Vec embeddings, we try to add emotional context to the existing embeddings of words. In short, we do this by first finding emotion words that are similar to the word we are trying to improve the embedding of, and then overlaying an emotion vector on top of the original Word2Vec vector. 
The Jupyter notebook source/sentimentAwareWE.ipynb shows the process, as well as the references.
