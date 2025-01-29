#named entity
import spacy
nlp = spacy.load("en_core_web_sm")
text=nlp(""""Apple is a leading technology company based in Cupertino, California. Founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne, Apple has revolutionized the tech industry with products like the iPhone, iPad, and MacBook. The company is also known for its software, including the iOS operating system and the iTunes software. In recent years, Apple has become one of the world's most valuable companies, with a market capitalization exceeding $2 trillion.""")
doc = nlp("Apple is looking at buying a startup in the U.K. for $1 billion.")
for ent in doc.ents:
    print(ent.text, ent.label_)  # Output: Apple ORG, U.K. GPE, $1 billion MONEY
