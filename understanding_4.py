import spacy

# Load the small English model in spaCy
nlp = spacy.load('en_core_web_sm')

# Input text
text = """Apple is a leading technology company based in Cupertino, California. 
Founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne, Apple has revolutionized 
the tech industry with products like the iPhone, iPad, and MacBook. The company is also 
known for its software, including the iOS operating system and the iTunes software. 
In recent years, Apple has become one of the world's most valuable companies, 
with a market capitalization exceeding $2 trillion."""

# Process the text with spaCy
doc = nlp(text)

# Extract sentences using `doc.sents`
sentences = [sent.text.strip() for sent in doc.sents]

# Print each sentence on a new line
print("Extracted Sentences:")
for idx, sent in enumerate(sentences, start=1):
    print(f"{idx}. {sent}")
