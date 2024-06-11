import nltk
nltk.download('stopwords')
# nltk.download('stopwords', download_dir="D:/mypython3/env3_10_11/nltk_data")

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import SnowballStemmer
import string


def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    if text is None:
        return ""
    
    preprocessed_text = remove_stopwords(text.replace("<br />", " ").lower())
    preprocessed_text = "".join([char for char in preprocessed_text if (char not in string.punctuation)])
    english_stemmer = SnowballStemmer('english')
    preprocessed_text = " ".join([english_stemmer.stem(i) for i in preprocessed_text.split()]) 

    return preprocessed_text