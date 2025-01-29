#stop words
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
text=""""Apple is a leading technology company based in Cupertino, California. Founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne, Apple has revolutionized the tech industry with products like the iPhone, iPad, and MacBook. The company is also known for its software, including the iOS operating system and the iTunes software. In recent years, Apple has become one of the world's most valuable companies, with a market capitalization exceeding $2 trillion."""
stopwords=list(STOP_WORDS)
print(stopwords)