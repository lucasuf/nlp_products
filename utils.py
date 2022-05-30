import heapq
import string
import pickle
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")


def process_text(text, to_str=False):
    """
    This method process the text by:
    - remove punctuation
    - remove stopwords and special characters
    - remove digits
    return a list of clean text words
    """
    special_formatted_chars = "–‘’"
    punctuation = string.punctuation + special_formatted_chars
    nopunc = [char for char in text if char not in punctuation]
    nopunc = "".join(nopunc)
    stopwords_list = (
        stopwords.words("english")
        + stopwords.words("spanish")
        + stopwords.words("portuguese")
    )
    clean_words = [
        word.lower()
        for word in nopunc.split()
        if (
            word.lower() not in stopwords_list
            and len(word) > 1
            and not bool(re.compile("\d|\@|\.|\(|\)|\/|\n").search(word))
        )
    ]
    if to_str:
        return " ".join(clean_words)
    return clean_words


def word_count(df_text):
    word_freq = {}
    for sentence in df_text:
        if isinstance(sentence, list):
            sentence = " ".join(sentence)
        try:
            tokens = word_tokenize(sentence)
        except Exception as e:
            print(f"error: {sentence}")
            raise Exception
        for token in tokens:
            if token not in word_freq.keys():
                word_freq[token] = 1
            else:
                word_freq[token] += 1
    return word_freq


def most_frequent_words(data, size):
    word_freq = word_count(data)
    return heapq.nlargest(size, word_freq, key=word_freq.get)


def pickle_dump(object, filename):
    with open(filename, "wb") as file_model:
        return pickle.dump(object, file_model)


def pickle_load(filename):
    with open(filename, "rb") as file_model:
        return pickle.load(file_model)
