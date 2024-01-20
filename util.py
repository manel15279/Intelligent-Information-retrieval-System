import indexer

def preprocess_query(query, Tokenize, PorterStemmer):

    if Tokenize:
        q = indexer.tokenization(query)
    else :
        q = query.split()
    if PorterStemmer:
        q = indexer.normalization_porter(q)
    else:
        q = indexer.normalization_lancaster(q)
    return q

def file(Tokenize, PorterStemmer):
    if Tokenize:
        if PorterStemmer:
            return "InverseTokenPorter.txt"
        else:
            return "InverseTokenLancaster.txt"
    else:
        if PorterStemmer:
            return "InverseSplitPorter.txt"
        else:
            return "InverseSplitLancaster.txt"
  