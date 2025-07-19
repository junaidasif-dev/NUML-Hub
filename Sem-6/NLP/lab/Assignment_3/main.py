from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
#nltk.download('punkt')

# Sample data
text = "Word embeddings are a type of word representation that allows words to be represented as vectors. These vectors capture semantic relationships between words."

# Preprocessing
tokens = word_tokenize(text.lower())

# Creating a list of sentences (each sentence is a list of words)
data = [tokens]

# Training the Word2Vec model
model = Word2Vec(sentences=data, vector_size=100, window=5, min_count=1, workers=4)

# Save model
model.save("word2vec.model")

# Example: Similarity
print("Similarity between 'word' and 'vectors':", model.wv.similarity('word', 'vectors'))

# Example: Word vector
print("Embedding for 'word':")
print(model.wv['word'])
