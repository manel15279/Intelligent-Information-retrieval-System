import nltk
import numpy as np
from collections import defaultdict
import preprocessing

def index(file_path, Inverse, Tokenize, PorterStemmer):
    """
    Index the documents in the given file.

    Args:
        file_path (str): Path to the file containing documents.
        Inverse (bool): Flag indicating whether to use inverse indexing.
        Tokenize (bool): Flag indicating whether to tokenize the text.
        PorterStemmer (bool): Flag indicating whether to use Porter stemming.

    Returns:
        dict: Indexed document data.
    """
    word_file_count = defaultdict(set)
    unique_document_numbers = set()
    d = {}
    dict = {}

    documents = preprocessing.extract_documents(file_path)

    for document in documents:
        file_id = document["number"]
        text = document["text"]
        unique_document_numbers.add(file_id)

        if Tokenize:
            tokens = preprocessing.tokenization(text)
        else:
            tokens = text.split()

        tokens_without_stopwords = preprocessing.stop_words(tokens)

        if PorterStemmer:
            normalized_words = preprocessing.normalization_porter(tokens_without_stopwords)
        else:
            normalized_words = preprocessing.normalization_lancaster(tokens_without_stopwords)

        words_frequency = nltk.FreqDist(normalized_words)

        for word in words_frequency.keys():
            if Inverse:
                if word in d:
                    d[word].append((file_id, words_frequency[word], max(list(words_frequency.values()))))
                else:
                    d[word] = [(file_id, words_frequency[word], max(list(words_frequency.values())))]
            else:
                if file_id in d:
                    d[file_id].append((word, words_frequency[word], max(list(words_frequency.values()))))
                else:
                    d[file_id] = [(word, words_frequency[word], max(list(words_frequency.values())))]
                word_file_count[word].add(file_id)

    for key1, values in d.items():
        for (key2, freq, max_freq) in values:
            if Inverse:
                value = (key2, freq, (freq / max_freq) * np.log10(((len(unique_document_numbers) / len(d[key1])) + 1)))
            else:
                value = (key2, freq, (freq / max_freq) * np.log10(((len(unique_document_numbers) / len(word_file_count[key2])) + 1)))
            if key1 in dict:
                dict[key1].append(value)
            else:
                dict[key1] = [value]

    output_file = preprocessing.file(Tokenize, PorterStemmer, Inverse)
    preprocessing.write_dict_to_file(dict, output_file)
            
    return dict
