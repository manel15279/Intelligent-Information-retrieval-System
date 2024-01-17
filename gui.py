import typing
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, uic
from PyQt5.QtWidgets import QWidget
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import re
import numpy as np 
from collections import defaultdict
from collections import Counter
from collections import OrderedDict
import matplotlib.pyplot as plt
from collections import defaultdict

#nltk.download()

files = os.listdir(os.path.abspath("Collection"))
queries_file = 'LISA COLLECTION\\Query.txt'
judgements_file = 'LISA COLLECTION\\LISARJ.NUM'


def tokenization(text):
    tokens = []
    ExpReg = nltk.RegexpTokenizer('(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*')
    tokens = ExpReg.tokenize(text)
    return tokens

def stop_words(tokens):
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    tokens_without_stopwords = []
    tokens_without_stopwords = [word for word in tokens if word.lower() not in nltk_stopwords]
    return tokens_without_stopwords

def normalization_porter(tokens): #stemming
    Porter = nltk.PorterStemmer()
    normalized_words = []
    normalized_words = [Porter.stem(word) for word in tokens]
    return normalized_words

def normalization_lancaster(tokens): #stemming
    Lancaster = nltk.LancasterStemmer()
    normalized_words = []
    normalized_words = [Lancaster.stem(word) for word in tokens]
    return normalized_words

def index(file_path, Inverse, Tokenize, PorterStemmer):
    word_file_count = defaultdict(set)
    unique_document_numbers = set() 
    d = {}
    dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('|', 1)
            
            if len(parts) == 2:
                file_id, text = parts
                file_id = file_id.strip()
                print(f"file_id={file_id}")
                unique_document_numbers.add(file_id)

                if Tokenize:
                    tokens = tokenization(text)
                else:
                    tokens = text.split()

                tokens_without_stopwords = stop_words(tokens)

                if PorterStemmer:
                    normalized_words = normalization_porter(tokens_without_stopwords)
                else:
                    normalized_words = normalization_lancaster(tokens_without_stopwords)

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
                    value = (key2, freq, (freq / max_freq) * np.log10(((len(unique_document_numbers) / len(d[key1]))+1)))
                else: 
                    value = (key2, freq, (freq / max_freq) * np.log10(((len(unique_document_numbers) / len(word_file_count[key2]))+1)))
                if key1 in dict:
                    dict[key1].append(value)
                else:
                    dict[key1] = [value]
                    
    return dict

def write_dict_to_file(dictionary, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for key, values in dictionary.items():
            for value in values:
                (files_list, freq, weight) = value
                file.write(f"{key} {files_list} {freq} {weight:.5f}\n")

def preprocess_query(query, Tokenize, PorterStemmer):

    if Tokenize:
        q = tokenization(query)
    else :
        q = query.split()
    if PorterStemmer:
        q = normalization_porter(q)
    else:
        q = normalization_lancaster(q)
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

    for doc, terms in terms_by_doc.items():
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

    for doc, terms in terms_by_doc.items():
        result_by_doc[doc] = terms / (sum_vi + weight_by_doc[doc] - terms)

    result_by_doc= sorted(result_by_doc.items(), key=lambda x: x[1], reverse=True)

    return result_by_doc

def vectorial_model(query, Tokenize, PorterStemmer, SP, cosine, jaccard):
        query = preprocess_query(query, Tokenize, PorterStemmer)
        file_path = file(Tokenize, PorterStemmer)
        if SP:
            result = scalar_product(query, file_path)
        else: 
            if cosine:
                result = cosine_measure(query, file_path)
            else:
                if jaccard:
                    result = jaccard_measure(query, file_path)

        return result

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
    query = preprocess_query(query, Tokenize, PorterStemmer)
    file_path = file(Tokenize, PorterStemmer)
    result = BM25(query, file_path, K, B)
        
    return result

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
    query = preprocess_query(query, Tokenize, PorterStemmer)
    file_path = file(Tokenize, PorterStemmer)

    result_dict = {}
    results = boolean_query_evaluation(query, file_path)
    
    if results is not None:
        for doc in results:
            result_dict[doc] = 'YES'
    else:
        result_dict = None

    return result_dict


def precision(relevant_docs, retrieved_docs, cutoff=None):
    total_retrieved = len(retrieved_docs[:cutoff])
    if total_retrieved == 0:
        return 0
    if cutoff == None:
        cutoff = total_retrieved
    relevant_retrieved = len(set(relevant_docs) & set(retrieved_docs[:cutoff]))
    return relevant_retrieved / cutoff

def recall(relevant_docs, retrieved_docs, cutoff=None):
    total_relevant = len(relevant_docs)
    if total_relevant == 0:
        return 0
    relevant_retrieved = len(set(relevant_docs) & set(retrieved_docs[:cutoff]))
    return relevant_retrieved / total_relevant

def f_score(precision_value, recall_value):
    if precision_value + recall_value > 0:
        return 2 * (precision_value * recall_value) / (precision_value + recall_value)
    else:
        return 0
    
def interpolate_precision(precision_values, recall_values, recall_levels):
    interpolated_precisions = []
    for level in recall_levels:
        max_precision = 0
        for i in range(len(recall_values)):
            if recall_values[i] >= level:
                max_precision = max(max_precision, precision_values[i])
        interpolated_precisions.append(max_precision)
    return interpolated_precisions

def plot_precision_recall_curve(relevant_docs, retrieved_docs):
    recall_levels = [i / 10 for i in range(11)]
    
    # Calculate precision at each recall level
    precision_values = [precision(relevant_docs, retrieved_docs, cutoff=int(len(retrieved_docs) * level)) for level in recall_levels]
    
    # Calculate interpolated precisions
    interpolated_precisions = interpolate_precision(precision_values, recall_levels, recall_levels)

    plt.plot(recall_levels, interpolated_precisions, marker='o', linestyle='-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend()
    plt.show()

def queries_tolist(queries_file):
    with open(queries_file, 'r') as file:
        queries = [line.strip().split('|', 1) for line in file]
    return queries

def judgements_tolist(judgements_file):
    query_relevant_docs = {}

    with open(judgements_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            tokens = line.split()
            query_id = int(tokens[0])
            num_relevant_docs = int(tokens[1])
            relevant_docs = [int(tokens[i]) for i in range(2, len(tokens))]
            query_relevant_docs[query_id] = {'num_relevant_docs': num_relevant_docs, 'relevant_docs': relevant_docs}

    return query_relevant_docs

def metrics(selected_docs, selected_relevant_docs):
    precision_value = precision(selected_relevant_docs, selected_docs)
    precision_5 = precision(selected_relevant_docs, selected_docs, 5)
    precision_10 = precision(selected_relevant_docs, selected_docs, 10)
    recall_value = recall(selected_relevant_docs, selected_docs)
    f_score_value = f_score(precision_value, recall_value) 

    return precision_value, precision_5, precision_10, recall_value, f_score_value


class MyGUI(QMainWindow):

    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi("TP.ui", self)
        self.search_button.clicked.connect(self.search)
        # Set column names
        self.tableWidget.setColumnCount(4)
        
        header = self.tableWidget.horizontalHeader()
        header.setDefaultAlignment(QtCore.Qt.AlignLeft)
        
        self.results = {}
        
        self.search_bar.setPlaceholderText("Search query...")
        if self.radioButton_indexs.isChecked():
            self.search_bar.textChanged.connect(self.query_results)
        self.search_bar.setFixedHeight(50) 

        self.query = ""
        
        self.show()
        
        
    def query_results(self):
        if self.radioButton_vsm.isChecked() or self.radioButton_proba.isChecked():
            self.display_search_results2(self.results)
        if  self.radioButton_bool.isChecked():
            self.display_search_results3(self.results)
        else:
            query = self.search_bar.text().strip().lower()
            filtered_results = {}
            self.tableWidget.setColumnWidth(0, 50)
            for key, values in self.results.items():
                if self.PorterStemmer:
                    Porter = nltk.PorterStemmer()
                    if str(key).startswith(Porter.stem(query.lower())):
                        filtered_results[key] = values
                    else:
                        filtered_values = [value for value in values if str(value[0]).startswith(Porter.stem(query.lower()))] 
                        if filtered_values:
                            filtered_results[key] = filtered_values
                else:
                    Lancaster = nltk.LancasterStemmer()
                    if str(key).startswith(Lancaster.stem(query.lower())):
                        filtered_results[key] = values
                    else:
                        filtered_values = [value for value in values if str(value[0]).startswith(Lancaster.stem(query.lower()))] 
                        if filtered_values:
                            filtered_results[key] = filtered_values

        self.display_search_results(filtered_results)
        

    def search(self):

        query = self.search_bar.text().strip().lower()
        self.PorterStemmer = self.checkBox_porter_stemmer.isChecked()
        self.Tokenize = self.checkBox_tokenization.isChecked()

        if self.radioButton_vsm.isChecked():
            self.sp = self.radioButton_SP.isChecked()
            self.cm = self.radioButton_CM.isChecked()
            self.jm = self.radioButton_JM.isChecked()
            self.results = vectorial_model(query, self.Tokenize, self.PorterStemmer, self.sp, self.cm, self.jm)

        else :
            if self.radioButton_proba.isChecked():
                self.K = self.lineEdit_K.text()
                self.B = self.lineEdit_B.text()
                self.results = probabilistic_model(query, self.Tokenize, self.PorterStemmer, float(self.K), float(self.B))
                
                self.display_search_results2(self.results)


            else:
                if self.radioButton_indexs.isChecked():
                    self.Inverse = self.radioButton_inverse.isChecked()
                    self.results = index(files, self.Inverse, self.Tokenize, self.PorterStemmer)
                    self.results = OrderedDict(sorted(self.results.items()))
                    
                    if self.Inverse:
                        if self.PorterStemmer:
                            if self.Tokenize:
                                filename = "InverseTokenPorter.txt"
                            else:
                                filename = "InverseSplitPorter.txt"
                        else:
                            if self.Tokenize:
                                filename = "InverseTokenLancaster.txt"
                            else:
                                filename = "InverseSplitLancaster.txt"
                    else:
                        if self.PorterStemmer:
                            if self.Tokenize:
                                filename = "DescripteurTokenPorter.txt"
                            else:
                                filename = "DescripteurSplitPorter.txt"
                        else:
                            if self.Tokenize:
                                filename = "DescripteurTokenLancaster.txt"
                            else:
                                filename = "DescripteurSplitLancaster.txt"
                    
                    write_dict_to_file(self.results, filename)
                    self.display_search_results(self.results)

                elif self.radioButton_bool.isChecked():
                    self.results = boolean_model(query, self.Tokenize, self.PorterStemmer)
                    
                    self.display_search_results3(self.results)
    
        

    def display_search_results(self, results):
        self.tableWidget.setHorizontalHeaderLabels(["Key", "Value", "Frequency", "Weight"])
        self.tableWidget.setRowCount(0)
        
        for key1, values in results.items():
            for value in values:
                rowPosition = self.tableWidget.rowCount()
                self.tableWidget.insertRow(rowPosition)
                self.tableWidget.setItem(rowPosition, 0, QTableWidgetItem(str(key1)))
                self.tableWidget.setItem(rowPosition, 1, QTableWidgetItem(str(value[0])))
                self.tableWidget.setItem(rowPosition, 2, QTableWidgetItem(str(value[1])))
                self.tableWidget.setItem(rowPosition, 3, QTableWidgetItem(f"{value[2]:.5f}"))
                
        total_width = self.tableWidget.viewport().width()
        column_width = int(total_width / self.tableWidget.columnCount())
        for column in range(self.tableWidget.columnCount()):
            self.tableWidget.setColumnWidth(column, column_width)
    
    def display_search_results2(self, results):
        self.tableWidget.setHorizontalHeaderLabels(["N Doc", "Relevance"])
        self.tableWidget.setRowCount(0)
        
        for (key, value) in results:
                rowPosition = self.tableWidget.rowCount()
                self.tableWidget.insertRow(rowPosition)
                self.tableWidget.setItem(rowPosition, 0, QTableWidgetItem(str(key)))
                self.tableWidget.setItem(rowPosition, 1, QTableWidgetItem(f"{float(value):.5f}"))
                
        total_width = self.tableWidget.viewport().width()
        column_width = int(total_width / self.tableWidget.columnCount())
        for column in range(self.tableWidget.columnCount()):
            self.tableWidget.setColumnWidth(column, column_width)

    def display_search_results3(self, results):
        self.tableWidget.setHorizontalHeaderLabels(["N Doc", "Relevance"])
        self.tableWidget.setRowCount(0)
        if self.results == None:
            rowPosition = self.tableWidget.rowCount()
            self.tableWidget.insertRow(rowPosition)
            self.tableWidget.setItem(rowPosition, 0, QTableWidgetItem('Invalid query !'))
        else:
            
            for key, value in results.items():
                    rowPosition = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(rowPosition)
                    self.tableWidget.setItem(rowPosition, 0, QTableWidgetItem(str(key)))
                    self.tableWidget.setItem(rowPosition, 1, QTableWidgetItem(str(value)))
                    
            total_width = self.tableWidget.viewport().width()
            column_width = int(total_width / self.tableWidget.columnCount())
            for column in range(self.tableWidget.columnCount()):
                self.tableWidget.setColumnWidth(column, column_width)


def main():
    app = QApplication([])
    window = MyGUI()
    app.exec_()


if __name__ == '__main__':
    main()