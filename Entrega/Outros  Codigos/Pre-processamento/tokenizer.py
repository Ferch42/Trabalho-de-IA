import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def tokenizer(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            if word in stopwords.words('english'):
                continue
            tokens.append(word.lower())
    return tokens

def tokenizer_lemmatizer(text):
    tokens = []
    wnl=WordNetLemmatizer()
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            if word in stopwords.words('english'):
                continue
            tokens.append(wnl.lemmatize(word.lower()))
    return tokens
