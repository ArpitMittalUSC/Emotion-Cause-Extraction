# Emotion-Cause-Extraction

### Some of the Required Libraries
Pandas
Numpy
Sklearn
benepar 
pycorenlp

### Download en_core_web_trf
python -m spacy download en_core_web_trf

### Sentiment/Emotion Aware Word Embeddings

Using initial Word2Vec embeddings, we try to add emotional context to the existing embeddings of words. In short, we do this by first finding emotion words that are similar to the word we are trying to improve the embedding of, and then overlaying an emotion vector on top of the original Word2Vec vector. 
The Jupyter notebook source/sentimentAwareWE.ipynb shows the process, as well as the references.
