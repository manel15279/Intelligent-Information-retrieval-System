import re
import util

def boolean_query(query):
    
    if isinstance(query, list):
        query = ' '.join(query)

    # reg expresssion 
    reg_exp = r'\b(?:((?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*)\b|and|or|not)\b'

    matches = re.findall(reg_exp, query)

    if not is_valid_boolean_query(matches):
        print("La requête n'est pas valide.")
        return None
    return matches

def is_valid_boolean_query(matches):
    if not matches:
        return False

    operators = {'and', 'or', 'not'}
    for match in matches:
        if match not in operators and not re.match(r'\b\w+\b', match):
            return False
    
    # operator order
    if matches[0] in operators-{'not'} or matches[-1] in operators:
        return False
    
    # NOT & term term 
    for i in range(len(matches) - 1):
        if matches[i] == 'not' and ((not matches[i + 1]) or (matches[i+1] in operators)):
            return False
        if  matches[i] not in operators and matches[i+1] not in operators:
            return False
        
    #  AND OR / OR AND
    for i in range(len(matches) - 2):
        if matches[i] in operators-{'not'} and matches[i + 1] in operators-{'not'}:
            return False
        
    return True

def boolean_query_evaluation(query, file_path):
    terms_and_operators = boolean_query(query)
    
    if terms_and_operators is None:
        return None
    else:
        result_set = set()

        with open(file_path, 'r') as file:
            lines = file.readlines()
            operator_stack = []

            for token in terms_and_operators:
                if token == 'not':
                    operator_stack.append('not')
                elif token == 'and':
                    operator_stack.append('and')
                elif token == 'or':
                    operator_stack.append('or')
                else:
                    term_results = set()
                    
                    for line in lines:
                        term, doc, _, _ = line.strip().split()
                        doc = int(doc)

                        if term == token:
                            term_results.add(doc)

                    if 'not' in operator_stack:
                        term_results = set() - term_results
                        operator_stack.remove('not')
                    if 'and' in operator_stack:
                        result_set = result_set.intersection(term_results)
                        operator_stack.remove('and')
                    elif 'or' in operator_stack:
                        result_set = result_set.union(term_results)
                        operator_stack.remove('or')
                    else:
                        result_set = term_results

        return result_set
    
def boolean_model(query, Tokenize, PorterStemmer):
    query = util.preprocess_query(query, Tokenize, PorterStemmer)
    file_path = util.file(Tokenize, PorterStemmer)

    result_dict = {}
    results = boolean_query_evaluation(query, file_path)
    
    if results is not None:
        for doc in results:
            result_dict[doc] = 'YES'
    else:
        result_dict = None

    return result_dict