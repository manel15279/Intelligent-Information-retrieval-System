import nltk
from collections import defaultdict
import indexer

def tokenization(text):
    """
    Tokenize the given text.

    Args:
        text (str): Input text to tokenize.

    Returns:
        list: List of tokens.
    """
    ExpReg = nltk.RegexpTokenizer(r'(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.,]\d+)?%?|\w+(?:[\-/]\w+)*')
    return ExpReg.tokenize(text)


def stop_words(tokens):
    """
    Remove stop words from the list of tokens.

    Args:
        tokens (list): List of tokens.

    Returns:
        list: List of tokens without stop words.
    """
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    return [word for word in tokens if word.lower() not in nltk_stopwords]


def normalization_porter(tokens):
    """
    Perform stemming using the Porter algorithm.

    Args:
        tokens (list): List of tokens.

    Returns:
        list: List of normalized tokens.
    """
    Porter = nltk.PorterStemmer()
    return [Porter.stem(word) for word in tokens]


def normalization_lancaster(tokens):
    """
    Perform stemming using the Lancaster algorithm.

    Args:
        tokens (list): List of tokens.

    Returns:
        list: List of normalized tokens.
    """
    Lancaster = nltk.LancasterStemmer()
    return [Lancaster.stem(word) for word in tokens]


def extract_documents(file_path):
    """
    Extract documents from the given file.

    Args:
        file_path (str): Path to the file containing documents.

    Returns:
        list: List of dictionaries representing documents.
    """
    documents = []
    current_document = {"number": None, "text": ""}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            if line.startswith("Document"):
                if current_document["number"] is not None:
                    documents.append(current_document.copy())
                current_document["number"] = int(line.split()[1])
                current_document["text"] = ""
            elif line:
                current_document["text"] += line + '\n'
    
    if current_document["number"] is not None:
        documents.append(current_document.copy())
    
    return documents


def write_dict_to_file(dictionary, filename):
    """
    Write dictionary content to a file.

    Args:
        dictionary (dict): Dictionary to write to the file.
        filename (str): Name of the output file.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        for key, values in dictionary.items():
            for value in values:
                files_list, freq, weight = value
                file.write(f"{key} {files_list} {freq} {weight:.5f}\n")


def preprocess_query(query, Tokenize, PorterStemmer):
    """
    Preprocess the query text.

    Args:
        query (str): Query text.
        Tokenize (bool): Flag indicating whether to tokenize the query.
        PorterStemmer (bool): Flag indicating whether to use Porter stemming.

    Returns:
        list: Preprocessed query tokens.
    """
    if Tokenize:
        q = tokenization(query)
    else:
        q = query.split()

    nltk_stopwords = nltk.corpus.stopwords.words('english')
    q = [word for word in q if word.lower() not in nltk_stopwords]

    if PorterStemmer:
        return normalization_porter(q)
    else:
        return normalization_lancaster(q)


def file(Tokenize, PorterStemmer, Inverse=None):
    """
    Generate file name based on Tokenize and PorterStemmer flags.

    Args:
        Inverse (bool): Flag indicating Descripteur or inverse file.
        Tokenize (bool): Flag indicating tokenization.
        PorterStemmer (bool): Flag indicating Porter stemming.

    Returns:
        str: File name.
    """
    if Inverse or Inverse==None:
        if Tokenize:
            if PorterStemmer:
                return "Inverses & Descriptors\InverseTokenPorter.txt"
            else:
                return "Inverses & Descriptors\InverseTokenLancaster.txt"
        else:
            if PorterStemmer:
                return "Inverses & Descriptors\InverseSplitPorter.txt"
            else:
                return "Inverses & Descriptors\InverseSplitLancaster.txt"
    else:
        if Tokenize:
            if PorterStemmer:
                return "Inverses & Descriptors\DescripteurTokenPorter.txt"
            else:
                return "Inverses & Descriptors\DescripteurTokenLancaster.txt"
        else:
            if PorterStemmer:
                return "Inverses & Descriptors\DescripteurSplitPorter.txt"
            else:
                return 'Inverses & Descriptors\DescripteurSplitLancaster.txt'
