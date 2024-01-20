import gradio as gr
import vectorialModel
import probabilisticModel
import booleanModel
import indexer
import evaluation
import pandas as pd

files = "LISA COLLECTION\\docs.txt"
queries_file = 'LISA COLLECTION\\Query.txt'
judgements_file = 'LISA COLLECTION\\LISA.REL'

def indexation(Inverse, Tokenize, PorterStemmer):
    dict = indexer.index(files, Inverse, Tokenize, PorterStemmer)
    

def queries_tolist(queries_file):
    with open(queries_file, 'r') as file:
        queries = [line.strip().split('|', 1) for line in file]
    print(queries)
    return queries

def judgements_tolist(file_path):
    relevant_refs_by_query = {}

    with open(file_path, 'r') as file:
        current_query_number = None
        current_relevant_refs = []
        relevant_refs_line = False

        for line in file:
            line = line.strip()

            if line.startswith("Query"):
                current_query_number = int(line.split()[1])
                relevant_refs_by_query[current_query_number] = []

            elif "Relevant Refs" in line:
                relevant_refs_line = True

            elif line.strip() and relevant_refs_line:
                current_relevant_refs = line.split()
                for val in current_relevant_refs[:-1]:
                    relevant_refs_by_query[current_query_number].append(val)
                relevant_refs_line = False

    return relevant_refs_by_query

def get_irs(irs):
    if irs == "Vectorial":
        return gr.Dropdown.update(choices=["Scalar Product", "Cosine Measure", "Jaccard Measure"])
    elif irs == "Boolean":
        return gr.Dropdown.update(choices=["Boolean Evaluation"])
    elif irs == "Probabilistic":
        return gr.Dropdown.update(choices=["BM25"])

def update_sliders(irs):
    if irs == "Probabilistic": return gr.Slider.update(visible=True)
    else: return gr.Slider.update(visible=False)

def set_queries():
    return gr.Dropdown.update(choices=queries_tolist(queries_file)[:, 1])


# fonction qui permet de définir la liste des documents en fonction du type de stemmer choisi
def update_docs(stem):
    return gr.Dropdown.update(choices=files.get_docs(stem))

# fonction qui permet de définir la liste des termes en fonction du type de stemmer choisi
def update_terms(stem):
    return gr.Dropdown.update(choices=files.get_terms(stem))

with gr.Blocks() as demo:
    
    gr.Markdown("<center><h1>Informarion Retrieval Systems</h1></center>")

    with gr.Tab("Indexation"):
        with gr.Column():

            with gr.Row():
                search = gr.Textbox(label="Search a query", placeholder='Search...')
                search_btn = gr.Button("Search") 

            with gr.Row():
                indexation_type = gr.Radio(label='Indexation', info="Select the file type :", choices=['Inverse', "Descriptor"])
                tokenization_type = gr.Radio(label='Tokenization', info="Select the tokenization method :", choices=['Inverse', "Descriptor"])
                stemmer_selector = gr.Radio(choices=["Porter", "Lancaster"], label="Stemming", info="Select a stemming method :")
                btn0 = gr.Button("Indexing") 

            with gr.Row():
                results = gr.DataFrame(label="Results")
                btn0.click(fn=indexation, inputs=[indexation_type, tokenization_type, stemmer_selector], outputs=results) 
                search_btn.click(fn=filter_results, inputs=[search, indexation_type, tokenization_type, stemmer_selector], outputs=results)

                
        with gr.Column(scale=1):
            with gr.Row():
                doc_selector = gr.Dropdown(label="Documents") # menu déroulant pour choisir les document à afficher
                stemmer_selector.change(update_docs, stemmer_selector, doc_selector) # on appelle la fonction update_docs qui va mettre à jour la liste des documents en fonction du type de stemmer choisi
                btn_doc = gr.Button("Show Document") # bouton pour afficher le fichier descripteur selon le document choisit
            with gr.Row():
                # dataframe pour afficher le fichier descripteur
                fichier_descripteur = gr.DataFrame(pd.DataFrame(columns=["Teme", "Freq", "Poids"]), label="Fichier Descripteur")
                # on appelle la fonction rech_fichier_descripteur qui va afficher le fichier descripteur selon le document choisi
                btn_doc.click(fn=files.rech_fichier_descripteur, inputs=[doc_selector, stemmer_selector], outputs=[fichier_descripteur])
        with gr.Column(scale=1):
            with gr.Row():
                term_selector = gr.Dropdown(label="Terms") # menu déroulant pour choisir les termes à afficher
                stemmer_selector.change(update_terms, stemmer_selector, term_selector) # on appelle la fonction update_terms qui va mettre à jour la liste des termes en fonction du type de stemmer choisi
                btn_term = gr.Button("Show Term") # bouton pour afficher le fichier inverse selon le terme choisi
            with gr.Row():
                # dataframe pour afficher le fichier inverse
                fichier_inverse = gr.DataFrame(pd.DataFrame(columns=["Teme", "Freq", "Poids"]), label="Fichier inverse")
                # on appelle la fonction rech_fichier_inverse qui va afficher le fichier inverse selon le terme choisi
                btn_term.click(fn=files.rech_fichier_inverse, inputs=[term_selector, stemmer_selector], outputs=[fichier_inverse])
    with gr.Tab("Appariement"):
        with gr.Row():
            with gr.Column(scale=1):
                # menu déroulant pour choisir le type de irs
                irs_selector = gr.Dropdown(["Vectorial", "Boolean", "Probabilistic", "Text Mining", "Deep Learning"], label="irs")

                function_selector = gr.Dropdown(label="Function", visible=True) # menu déroulant pour choisir le type de fonction de similarité
                irs_selector.change(get_function, irs_selector, function_selector) # on appelle la fonction get_function qui va mettre à jour la liste des fonctions en fonction du type de irs choisi

                query_selector = gr.Dropdown(label="Query", visible=True) # menu déroulant pour choisir la requête à afficher
                irs_selector.change(get_set_queries, irs_selector, query_selector) # on appelle la fonction get_set_queries qui va mettre à jour la liste des requêtes en fonction du type de irs choisi

                b_value = gr.Slider(0.5, 0.75, 0.6, label="b value", visible=False) # slider pour choisir la valeur de b
                irs_selector.change(update_sliders, irs_selector, b_value) # on appelle la fonction update_sliders qui va mettre à jour l'affichage du slider en fonction du type de irs choisi
                k_value = gr.Slider(1.2, 2, 1.5, label="k value", visible=False) # slider pour choisir la valeur de k
                irs_selector.change(update_sliders, irs_selector, k_value) # on appelle la fonction update_sliders qui va mettre à jour l'affichage du slider en fonction du type de irs choisi

                radius = gr.Slider(0, 1, 0.6, label="Radius", visible=False) # slider pour choisir la valeur du rayon
                irs_selector.change(update_dbscan_rad, irs_selector, radius) # on appelle la fonction update_dbscan_rad qui va mettre à jour l'affichage du slider en fonction du type de irs choisi
                min_neighbours = gr.Number(value= 200, label="Min Neighbours", visible=False) # nombre pour choisir le nombre de voisins minimum
                irs_selector.change(update_dbscan_minn, irs_selector, min_neighbours) # on appelle la fonction update_dbscan_minn qui va mettre à jour l'affichage de la nombrebox en fonction du type de irs choisi

                btn1 = gr.Button("Compute") # bouton pour lancer le calcul de la similarité
            with gr.Column(scale=3):
                df_out = gr.DataFrame(pd.DataFrame(columns=["Document", "Score"]), label="Results") # dataframe pour afficher les résultats
                # on appelle la fonction apply_irs qui va calculer la similarité selon le type de irs choisi
                btn1.click(fn=irs.apply_irs, inputs=[stemmer_selector, query_selector, function_selector, b_value, k_value, radius, min_neighbours], outputs=df_out)
            with gr.Column(scale=1):
                p = gr.Number(label="Precision") # nombrebox pour afficher la précision
                p5 = gr.Number(label="P@5") # nombrebox pour afficher la précision à 5
                irs_selector.change(update_displayed_metrics, irs_selector, p5) # on appelle la fonction update_displayed_metrics qui va mettre à jour l'affichage des nombrebox en fonction du type de irs choisi
                p10 = gr.Number(label="P@10") # nombrebox pour afficher la précision à 10
                irs_selector.change(update_displayed_metrics, irs_selector, p10) # on appelle la fonction update_displayed_metrics qui va mettre à jour l'affichage des nombrebox en fonction du type de irs choisi
                metrics_out = [p, p5, p10, gr.Number(label="Recall"), gr.Number(label="F measure"), gr.Plot()] # liste des nombrebox pour afficher les métriques
                btn2 = gr.Button("Compute Metrics") # bouton pour lancer le calcul des métriques
                irs_selector.change(update_metrics_button, irs_selector, btn2) # on appelle la fonction update_metrics_button qui va mettre à jour l'affichage du bouton en fonction du type de irs choisi
                btn2.click(fn=irs.metrics, inputs=[query_selector, df_out], outputs=metrics_out) # on appelle la fonction metrics qui va calculer les métriques selon le type de irs choisi

if __name__ == "__main__":
    demo.launch()