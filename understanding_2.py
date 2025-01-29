#part of speech
import spacy
nlp = spacy.load("en_core_web_sm")
text=nlp(""""Apple is a leading technology company based in Cupertino, California. Founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne, Apple has revolutionized the tech industry with products like the iPhone, iPad, and MacBook. The company is also known for its software, including the iOS operating system and the iTunes software. In recent years, Apple has become one of the world's most valuable companies, with a market capitalization exceeding $2 trillion.""")
for token in text:
    print(token.text,token.pos_)