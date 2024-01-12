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

#nltk.download()

files = os.listdir(os.path.abspath("Collection"))


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


def index(files, Inverse, Tokenize, PorterStemmer):
    word_file_count = defaultdict(set)
    d = {}
    dict = {}
    for filename in files: 
        with open(os.path.join("Collection", filename), "r") as file:
            text = file.read()

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

            file_id = int(re.search(r'\d+', os.path.basename(filename)).group())

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
                    value = (key2, freq, (freq / max_freq) * np.log10(((len(files) / len(d[key1]))+1)))
                else: 
                    value = (key2, freq, (freq / max_freq) * np.log10(((len(files) / len(word_file_count[key2]))+1)))
                if key1 in dict:
                    dict[key1].append(value)
                else:
                    dict[key1] = [value]
                    
    return dict


def write_dict_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        for key, values in dictionary.items():
            for value in values:
                (files_list, freq, weight) = value
                file.write(f"{key} {files_list} {freq} {weight:.5f}\n")

def write_relevance_to_file(dict, filename):
    with open(filename, 'w') as file:
        if dict == None:
            file.write("Invalid query !")
        else:
            for key, value in dict.items():
                if type(value) == float:
                    file.write(f"{key} {value:.5f}\n")
                else:
                    file.write(f"{key} {value}\n")

def scalar_product(query, doc, index, dict):
    RSV = 0

    for term in dict.keys():
        term_query_weight = 1 if term in query else 0
        weight = [t[2] for t in dict[term] if t[0] == index]
        if not weight:
            weight = 0
        else:
            weight = weight[0]
        term_weight = term_query_weight * weight
        RSV += term_weight

    return RSV

def cosine_measure(query, doc, index, dict):
    RSV = 0
    sum_vi = 0
    sum_wi = 0

    for term in dict.keys():
        term_query_weight = 1 if term in query else 0
        weight = [t[2] for t in dict[term] if t[0] == index]
        if not weight:
            weight = 0
        else:
            weight = weight[0]
        term_weight = term_query_weight * weight
        RSV += term_weight
        sum_wi += (weight**2)
        sum_vi += (term_query_weight**2)
    
    if np.sqrt(sum_vi) == 0 or np.sqrt(sum_wi) == 0:
        return 0.0

    res = RSV / (np.sqrt(sum_vi) * np.sqrt(sum_wi))
    return res

def jaccard_measure(query, doc, index, dict):
    RSV = 0
    sum_vi = 0
    sum_wi = 0

    for term in dict.keys():
        term_query_weight = 1 if term in query else 0
        weight = [t[2] for t in dict[term] if t[0] == index]
        if not weight:
            weight = 0
        else:
            weight = weight[0]
        term_weight = term_query_weight * weight
        RSV += term_weight
        sum_wi += weight**2
        sum_vi += term_query_weight**2

    if (sum_vi + sum_wi - RSV) == 0:
        return 0.0

    res = RSV / ((sum_vi + sum_wi) - RSV)
    return res

def preprocessing(files, Tokenize, PorterStemmer):
    docs = {}

    for filename in files: 
        with open(os.path.join("Collection", filename), "r") as file:
            text = file.read()

            if Tokenize:
                tokens = tokenization(text)
            else:
                tokens = text.split()

            tokens_without_stopwords = stop_words(tokens)

            if PorterStemmer:
                normalized_words = normalization_porter(tokens_without_stopwords)
            else:
                normalized_words = normalization_lancaster(tokens_without_stopwords)
            
            file_id = int(re.search(r'\d+', os.path.basename(filename)).group())
            docs[file_id] = normalized_words
    return docs

def vectorial_model(query, files, Tokenize, PorterStemmer, SP, cosine, jaccard):
    docs = preprocessing(files, Tokenize, PorterStemmer)
    query = preprocess_query(query, Tokenize, PorterStemmer)
    dict = index(files, True, Tokenize, PorterStemmer)
    ranking = OrderedDict()
    N = len(docs)

    for id, doc in docs.items():
        if SP:
            RSV = scalar_product(query, doc, id, dict)
            if RSV != 0.0:
                ranking[id] = RSV
        else: 
            if cosine:
                RSV = cosine_measure(query, doc, id, dict)
                if RSV != 0.0:
                    ranking[id] = RSV
            else:
                if jaccard:
                    RSV = jaccard_measure(query, doc, id, dict)
                    if RSV != 0.0:
                        ranking[id] = RSV

        ranking = OrderedDict(sorted(ranking.items(), key=lambda x: x[1], reverse=True))

    return ranking

def BM25(query, dict, docs, docs_id, K, B):
    ranking = OrderedDict()
    N = len(docs_id)
    
    s = 0
    for doc in docs.values():
        s += len(doc)
    avdl = s / N

    for index, doc in docs.items():
        RSV = 0 
        for term in query:
            freq = doc.count(term)
            nbr_docs = len(dict[term])
            term_weight = (freq / (K * ((1 - B) + B * (len(doc) / avdl)) + freq))
            RSV += (term_weight * np.log10(((N - nbr_docs + 0.5) / (nbr_docs + 0.5))))

        if RSV != 0.0:
            ranking[index] = RSV

    ranking = OrderedDict(sorted(ranking.items(), key=lambda x: x[1], reverse=True))
    return ranking

def probabilistic_model(query, files, Tokenize, PorterStemmer, K, B):
    query = preprocess_query(query, Tokenize, PorterStemmer)
    docs = preprocessing(files, Tokenize, PorterStemmer)
    docs_id = get_docs_ids(files)
    dict = index(files, True, Tokenize, PorterStemmer)
    results = results = BM25(query, dict, docs, docs_id, K, B)

    return results

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

def get_docs_ids(files):
    docs_id = []
    for filename in files: 
        file_id = int(re.search(r'\d+', os.path.basename(filename)).group())
        docs_id.append(file_id)
    return docs_id

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

def boolean_query_evaluation(query, dict, docs_id):

    terms_and_operators = boolean_query(query)
    if terms_and_operators == None:
        return None
    else:
        
        result_set = set(docs_id)

        operator_stack = []

        for token in terms_and_operators:
            if token == 'not':
                operator_stack.append('not')
            elif token == 'and':
                operator_stack.append('and')
            elif token == 'or':
                operator_stack.append('or')
            else:
                term_results = set(tup[0] for tup in dict[token]) if token in dict else set()
                if 'not' in operator_stack:
                    term_results = set(docs_id) - term_results
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
    
def boolean_model(query, files, Tokenize, PorterStemmer):
    result_dict = {}
    query = preprocess_query(query, Tokenize, PorterStemmer)
    print(query)
    docs_id = get_docs_ids(files)
    dict = index(files, True, Tokenize, PorterStemmer)
    results = boolean_query_evaluation(query, dict, docs_id)
    if results != None:
        for doc in docs_id:
            if doc in results:
                result_dict[doc] = 'YES'
            else:
                result_dict[doc] = 'NO'
    else:
        result_dict = None

    return result_dict



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
            self.results = vectorial_model(query, files, self.Tokenize, self.PorterStemmer, self.sp, self.cm, self.jm)
            if self.sp:
                filename = "VsmScalarProduct.txt"
            else:
                if self.cm:
                    filename = "VsmCosineMeasure.txt"
                else:
                    if self.jm:
                        filename ="VsmJaccardMeasure.txt"

            write_relevance_to_file(self.results, filename)
            self.display_search_results2(self.results)

        else :
            if self.radioButton_proba.isChecked():
                self.K = self.lineEdit_K.text()
                self.B = self.lineEdit_B.text()
                self.results = probabilistic_model(query, files, self.Tokenize, self.PorterStemmer, float(self.K), float(self.B))
                filename = "PmBM25.txt"
                write_relevance_to_file(self.results, filename)
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
                    self.results = boolean_model(query, files, self.Tokenize, self.PorterStemmer)
                    filename = "BM.txt"
                    write_relevance_to_file(self.results, filename)
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
        
        for key, value in results.items():
                rowPosition = self.tableWidget.rowCount()
                self.tableWidget.insertRow(rowPosition)
                self.tableWidget.setItem(rowPosition, 0, QTableWidgetItem(str(key)))
                self.tableWidget.setItem(rowPosition, 1, QTableWidgetItem(f"{value:.5f}"))
                
        total_width = self.tableWidget.viewport().width()
        column_width = int(total_width / self.tableWidget.columnCount())
        for column in range(self.tableWidget.columnCount()):
            self.tableWidget.setColumnWidth(column, column_width)

    def display_search_results3(self, results):
        self.tableWidget.setHorizontalHeaderLabels(["N Doc", "YES/NO"])
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