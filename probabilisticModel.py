import util

def n_docs_terms(file_path, query):
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
    query = util.preprocess_query(query, Tokenize, PorterStemmer)
    file_path = util.file(Tokenize, PorterStemmer)
    result = BM25(query, file_path, K, B)
        
    return result