import numpy as np
import preprocessing


def n_docs_terms(file_path, query):
    """
    Count the number of documents containing each term in the query.

    Args:
        file_path (str): Path to the file containing term-document frequencies.
        query (list): List of query terms.

    Returns:
        dict: Dictionary containing term-document counts.
    """
    documents_containing_terms = {}

    with open(file_path, 'r') as file:
        for line in file:
            current_term, _, _, _ = line.strip().split()

            if current_term in query:
                if current_term not in documents_containing_terms:
                    documents_containing_terms[current_term] = 1
                else:
                    documents_containing_terms[current_term] += 1
            elif len(documents_containing_terms) == len(query):
                # If we have encountered all terms and the next one is different, stop the loop
                break
    return documents_containing_terms


def BM25(query, file_path, K, B):
    """
    Calculate BM25 scores for documents based on the query.

    Args:
        query (list): List of query terms.
        file_path (str): Path to the file containing term-document frequencies.
        K (float): BM25 constant K.
        B (float): BM25 constant B.

    Returns:
        list: List of tuples containing (document_id, score) pairs, sorted by score in descending order.
    """
    dl = {}
    result = {}
    vocab_len = 0

    with open(file_path, 'r') as file:
        for line in file:
            _, doc_id, freq, _ = line.strip().split()
            dl[doc_id] = dl.get(doc_id, 0) + int(freq)
            vocab_len += int(freq)

    N = len(dl)
    avdl = vocab_len / N
    ni = n_docs_terms(file_path, query)

    with open(file_path, 'r') as file:
        for line in file:
            term, doc, freq, _ = line.split()
            freq = int(freq)
            if term in query:
                if doc in result:
                    result[doc] += ((freq / (K * ((1 - B) + B * (dl[doc] / avdl)) + freq)) * np.log10(((N - ni[term] + 0.5) / (ni[term] + 0.5))))
                else:
                    result[doc] = ((freq / (K * ((1 - B) + B * (dl[doc] / avdl)) + freq)) * np.log10(((N - ni[term] + 0.5) / (ni[term] + 0.5))))
                    
    result = sorted(result.items(), key=lambda x: x[1], reverse=True)

    return result


def probabilistic_model(query, Tokenize, PorterStemmer, K, B):
    """
    Perform probabilistic retrieval model.

    Args:
        query (str): Query text.
        Tokenize (bool): Flag indicating whether to tokenize the query.
        PorterStemmer (bool): Flag indicating whether to use Porter stemming.
        K (float): BM25 constant K.
        B (float): BM25 constant B.

    Returns:
        list: List of tuples containing (document_id, score) pairs, sorted by score in descending order.
    """
    query = preprocessing.preprocess_query(query, Tokenize, PorterStemmer)
    file_path = preprocessing.file(Tokenize, PorterStemmer)
    result = BM25(query, file_path, K, B)
        
    return result
