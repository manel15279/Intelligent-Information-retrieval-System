import matplotlib.pyplot as plt
import pandas as pd

def precision(relevant_docs, selected_docs, i=None): 
    """
    Calculate precision.

    Args:
        relevant_docs (list): List of relevant documents.
        selected_docs (list): List of selected documents.
        i (int, optional): Cutoff value for precision. Defaults to None for average precision.

    Returns:
        float: Precision value.
    """
    selected_relevant_docs_i = [doc for doc in relevant_docs if doc in selected_docs[:i]]

    if i is None:  # Average precision 
        if len(selected_docs)==0:
            return 0.0
        else:
            return len(selected_relevant_docs_i) / len(selected_docs)
    else:  # Precision @i
        return len(selected_relevant_docs_i) / i

def recall(relevant_docs, retrieved_docs, cutoff=None):
    """
    Calculate recall.

    Args:
        relevant_docs (list): List of relevant documents.
        retrieved_docs (list): List of retrieved documents.
        cutoff (int, optional): Cutoff value for recall. Defaults to None.

    Returns:
        float: Recall value.
    """
    total_relevant = len(relevant_docs)
    if total_relevant == 0:
        return 0
    relevant_retrieved = len(set(relevant_docs) & set(retrieved_docs[:cutoff]))
    return relevant_retrieved / total_relevant

def f_score(precision_value, recall_value):
    """
    Calculate F-score.

    Args:
        precision_value (float): Precision value.
        recall_value (float): Recall value.

    Returns:
        float: F-score value.
    """
    if precision_value + recall_value > 0:
        return 2 * (precision_value * recall_value) / (precision_value + recall_value)
    else:
        return 0
    
def plot_precision_recall_curve(selected_docs, relevant_docs):
    """
    Plot precision-recall curve and save as PNG image.

    Args:
        selected_docs (list): List of selected documents.
        relevant_docs (list): List of relevant documents.
    """
    l = len(selected_docs) 
    selected_relevant_docs = [doc for doc in relevant_docs if doc in selected_docs]
    k = len(selected_relevant_docs) 

    rp = [] 

    for i in range(1,l):
        pi = precision(relevant_docs, selected_docs, i) 
        selected_relevant_docs_i = [doc for doc in relevant_docs if doc in selected_docs[:i]]
        rp.append([pi, len(selected_relevant_docs_i)/k])

    rp = pd.DataFrame(rp, columns=["Precision", "Recall"]) 

    rpi = [] 
    j = 0.0
    while j <= 1.0:
        p_max = rp.loc[rp["Recall"] >= j]["Precision"].max() 
        rpi.append([j, p_max])
        j += 0.1

    rpi = pd.DataFrame(rpi, columns=["Recall", "Precision"]) 
    plt.figure() 
    plt.title("Precision-Recall Curve") 
    plt.xlabel("Recall") 
    plt.ylabel("Precision")
    plt.grid(True)
    plt.plot(rpi["Recall"], rpi["Precision"])
    image_path = 'plots\precision_recall_curve.png'
    plt.savefig(image_path, format='png')
    plt.close()



def get_metrics(selected_docs, selected_relevant_docs, relevant_docs):
    """
    Calculate precision, recall, and F-score based on selected and relevant documents.
    """
    precision_value = precision(selected_relevant_docs, selected_docs)
    precision_5 = precision(relevant_docs, selected_docs, 5)
    precision_10 = precision(relevant_docs, selected_docs, 10)
    recall_value = recall(selected_relevant_docs, selected_docs)
    f_score_value = f_score(precision_value, recall_value)

    return precision_value, precision_5, precision_10, recall_value, f_score_value
