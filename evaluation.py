import matplotlib.pyplot as plt
import pandas as pd


def precision(relevant_docs, selected_docs, i): 
    selected_relevant_docs_i = [doc for doc in relevant_docs if doc in selected_docs[:i]]
    return len(selected_relevant_docs_i) / i

def precision_avg(relevant_docs, retrieved_docs, cutoff=None):
    total_retrieved = len(retrieved_docs[:cutoff])
    if total_retrieved == 0:
        return 0
    if cutoff == None:
        cutoff = total_retrieved
    relevant_retrieved = len(set(relevant_docs))
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
    
def plot_precision_recall_curve(selected_docs, relevant_docs):
    l = len(selected_docs) 
    selected_relevant_docs = [doc for doc in relevant_docs if doc in selected_docs]
    k = len(selected_relevant_docs) 

    rp = [] 

    for i in range(1,l):
        pi = precision(relevant_docs, selected_docs, i) 
        selected_relevant_docs_i = [doc for doc in relevant_docs if doc in selected_docs[:i]]
        rp.append([pi, len(selected_relevant_docs_i)/k])

    rp = pd.DataFrame(rp, columns=["Precision", "Rappel"]) 

    rpi = [] 
    j = 0.0
    while j <= 1.0:
        p_max = rp.loc[rp["Rappel"] >= j]["Precision"].max() 
        rpi.append([j, p_max])
        j += 0.1

    rpi = pd.DataFrame(rpi, columns=["Rappel", "Precision"]) 
    plt.figure() 
    plt.title("Courbe Rappel/Précision") 
    plt.xlabel("Rappel") 
    plt.ylabel("Précision")
    plt.grid(True)
    plt.plot(rpi["Rappel"], rpi["Precision"])
    image_path = 'precision_recall_curve.png'
    plt.savefig(image_path, format='png')
    plt.close()


def metrics(selected_docs, selected_relevant_docs):
    precision_value = precision_avg(selected_relevant_docs, selected_docs)
    precision_5 = precision_avg(selected_relevant_docs, selected_docs, 5)
    precision_10 = precision_avg(selected_relevant_docs, selected_docs, 10)
    recall_value = recall(selected_relevant_docs, selected_docs)
    f_score_value = f_score(precision_value, recall_value) 

    return precision_value, precision_5, precision_10, recall_value, f_score_value
