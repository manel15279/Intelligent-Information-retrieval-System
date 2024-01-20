import nltk
import numpy as np 
from collections import defaultdict


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

def extract_documents(file_path):
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
    with open(filename, 'w', encoding='utf-8') as file:
        for key, values in dictionary.items():
            for value in values:
                (files_list, freq, weight) = value
                file.write(f"{key} {files_list} {freq} {weight:.5f}\n")

def index(file_path, Inverse, Tokenize, PorterStemmer):
            word_file_count = defaultdict(set)
            unique_document_numbers = set() 
            d = {}
            dict = {}
            
            documents = extract_documents(file_path)

            for document in documents:
                file_id = document["number"]
                text = document["text"]
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
            
            if Inverse:
                if Tokenize:
                    if PorterStemmer:
                        write_dict_to_file(dict, "InverseTokenPorter.txt")
                    else:
                        write_dict_to_file(dict, "InverseTokenLancaster.txt") 
                else:
                    if PorterStemmer:
                        write_dict_to_file(dict, "InverseSplitPorter.txt") 
                    else:
                        write_dict_to_file(dict, "InverseSplitLancaster.txt") 
            else:
                if Tokenize:
                    if PorterStemmer:
                        write_dict_to_file(dict, "DescripteurTokenPorter.txt") 
                    else:
                        write_dict_to_file(dict, "DescripteurTokenLancaster.txt") 
                else:
                    if PorterStemmer:
                        write_dict_to_file(dict, "DescripteurSplitPorter.txt") 
                    else:
                        write_dict_to_file(dict, "DescripteurSplitLancaster.txt") 
                            
            return dict
