import math
import util

def scalar_product(query, file_path):
    terms_by_doc = {}
    with open(file_path, 'r') as file:
        for line in file:
            term, doc_id, freq, weight = line.split()
            if term in query:
                weight = float(weight) 

                if doc_id in terms_by_doc:
                    terms_by_doc[doc_id] += weight
                else:
                    terms_by_doc[doc_id] = weight

    terms_by_doc = sorted(terms_by_doc.items(), key=lambda x: x[1], reverse=True)

    return terms_by_doc

def cosine_measure(query, file_path):
    terms_by_doc = scalar_product(query, file_path)
    weight_by_doc = {}
    result_by_doc = {}
    sum_vi = len(query)

    with open(file_path, 'r') as file:
        for line in file:
            term, doc, freq, weight = line.split()
            weight = float(weight) 
  
            if doc in weight_by_doc:
                weight_by_doc[doc] += weight**2
            else:
                weight_by_doc[doc] = weight**2

    sum_vi = math.sqrt(sum_vi)
    
    for doc, sum_squared in weight_by_doc.items():
        weight_by_doc[doc] = math.sqrt(sum_squared)

    for (doc, terms) in terms_by_doc:
        result_by_doc[doc] = terms / (sum_vi * weight_by_doc[doc])

    result_by_doc= sorted(result_by_doc.items(), key=lambda x: x[1], reverse=True)

    return result_by_doc

def jaccard_measure(query, file_path):
    terms_by_doc = scalar_product(query, file_path)
    weight_by_doc = {}
    result_by_doc = {}
    sum_vi = len(query)

    with open(file_path, 'r') as file:
        for line in file:
            term, doc, freq, weight = line.split()
            weight = float(weight)  
  
            if doc in weight_by_doc:
                weight_by_doc[doc] += weight**2
            else:
                weight_by_doc[doc] = weight**2

    for (doc, terms) in terms_by_doc:
        result_by_doc[doc] = terms / (sum_vi + weight_by_doc[doc] - terms)

    result_by_doc= sorted(result_by_doc.items(), key=lambda x: x[1], reverse=True)

    return result_by_doc

def vectorial_model(query, Tokenize, PorterStemmer, SP, cosine, jaccard):
        query = util.preprocess_query(query, Tokenize, PorterStemmer)
        file_path = util.file(Tokenize, PorterStemmer)
        if SP:
            result = scalar_product(query, file_path)
        else: 
            if cosine:
                result = cosine_measure(query, file_path)
            else:
                if jaccard:
                    result = jaccard_measure(query, file_path)

        return result