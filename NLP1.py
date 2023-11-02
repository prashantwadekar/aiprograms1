# -*- coding: utf-8 -*-
"""
In this example, we're using NLTK to preprocess a sample text for NLP. 
We start by downloading necessary resources such as the punkt tokenizer, WordNet lemmatizer, and stopwords. 
We then define a sample text, convert it to lowercase, tokenize it using NLTK's word_tokenize function, 
remove stopwords, lemmatize the remaining tokens, and count the frequency of each word using 
the Counter function from the collections module. 
Finally, we print the most common words in the text along with their frequency counts.
"""

# https://www.analyticsvidhya.com/blog/2021/07/getting-started-with-nlp-using-nltk-library/

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from nltk.probability import FreqDist
from nltk.corpus import wordnet

# download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# define a sample text
text = "Natural language processing (NLP) is a field of computer science, artificial intelligence, and computational linguistics concerned with the interactions between computers and human languages."

# convert text to lowercase
text = text.lower()

# tokenize text - splitting the text data into individual words
tokens = word_tokenize(text)

# remove stop words
stop_words = set(stopwords.words('english'))
tokens = [token for token in tokens if not token in stop_words]

# lemmatize tokens - reduce the word to their root word.
# It uses vocabulary and morphological analysis to transform a word into a root word.
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(token) for token in tokens]

# count word frequency
word_freq = Counter(tokens)

# print most common words
print(word_freq.most_common())

#Frequency distribution of word
freq_dist_of_words = FreqDist(tokens)
print(freq_dist_of_words)
freq_dist_of_words.most_common(5)
freq_dist_of_words.plot(30,cumulative=False)

# Parts of Speech (PoS) tagging
# To identify nouns, verbs, adjectives, adverbs, etc., and tag each word
nltk.pos_tag(tokens, tagset='universal')

# Named Entity Recognition
# to identify names of organizations, people, and geographic locations in the text and tag them to the text
tags = nltk.pos_tag(tokens, tagset='universal')
entities = nltk.chunk.ne_chunk(tags, binary=False)
print(entities)

# WordNet is a huge collection of words with meanings just like a traditional dictionary, used to generate synonyms, antonyms of words.
synonym = wordnet.synsets("AI")
print(synonym)
print(synonym[1].definition())
print(synonym[1].examples())



