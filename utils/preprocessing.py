import re
import nltk
from nltk.corpus import stopwords

stopword_list = set(stopwords.words('english'))

def clean_text(text):
    if text is None:
        return ""
    
    text = text.lower()
    
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def truncate_text(text, max_words=300):
    if text is None:
        return ""

    words = text.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words])
    return text


def remove_stopwords(text, stopword_list=stopword_list):
    if text is None:
        return ""
    
    words = text.split()
    filtered = [word for word in words if word not in stopword_list]
    return ' '.join(filtered)

def preprocess_pipeline(text, max_words=300, stopword_list=stopword_list):
    if text is None:
        return ""

    text = clean_text(text)
    text = truncate_text(text, max_words=max_words)
    text = remove_stopwords(text, stopword_list=stopword_list)
    
    return text