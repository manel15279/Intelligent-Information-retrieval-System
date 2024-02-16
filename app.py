from PyQt5.QtWidgets import *
from PyQt5 import QtCore, uic
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
import nltk
import metrics
import indexer
import vectorialModel
import booleanModel
import probabilisticModel
import loader
import os
import preprocessing

#nltk.download()

FILES_PATH = "LISA COLLECTION"
DOCS_FILE = os.path.join(FILES_PATH, 'docs.txt')
QUERIES_FILE = os.path.join(FILES_PATH, 'Query.txt')
JUDGEMENTS_FILE = os.path.join(FILES_PATH, 'LISA.REL')


class MyGUI(QMainWindow):

    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi("TP.ui", self)
        self.indexation_btn.clicked.connect(self.indexation)
        # Set column names
        self.tableWidget.setColumnCount(4)

        header = self.tableWidget.horizontalHeader()
        header.setDefaultAlignment(QtCore.Qt.AlignLeft)

        self.results = {}
        self.search_bar.setPlaceholderText("Search query...")
        self.search_button.clicked.connect(self.handle_search)
        self.search_bar.setFixedHeight(50)
        self.indexation_btn.setFixedWidth(110)

        self.query = ""

        self.show()

    def handle_search(self):
        if self.checkBox_queries_dataset.isChecked():
            self.search()
        else:
            self.display_query_results()
        

    def display_query_results(self):
        """
        Display query results based on selected options.
        """
        # Get selected options
        self.Inverse = self.radioButton_inverse.isChecked()
        self.PorterStemmer = self.checkBox_porter_stemmer.isChecked()
        self.Tokenize = self.checkBox_tokenization.isChecked()
        query = self.search_bar.text().strip().lower().split()  # Split query into terms
        filtered_results = {}
        self.tableWidget.setColumnWidth(0, 50)
        
        file_name = preprocessing.file(self.Tokenize, self.PorterStemmer, self.Inverse)

        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file:
                if not self.Inverse:
                    file_number, term, frequency, weight = line.strip().split()
                    if any(term.lower().startswith(q) for q in query):
                        if file_number in filtered_results:
                            filtered_results[file_number].append((term, int(frequency), float(weight)))
                        else:
                            filtered_results[file_number] = [(term, int(frequency), float(weight))]
                else:
                    term, file_number, frequency, weight = line.strip().split()
                    if any(term.lower().startswith(q) for q in query):
                        if term in filtered_results:
                            filtered_results[term].append((int(file_number), int(frequency), float(weight)))
                        else:
                            filtered_results[term] = [(int(file_number), int(frequency), float(weight))]
        
        # Display filtered results
        self.display_search_results(filtered_results)

    def indexation(self):
            """
            Perform document indexation based on selected options.
            """
            self.Inverse = self.radioButton_inverse.isChecked()
            self.PorterStemmer = self.checkBox_porter_stemmer.isChecked()
            self.Tokenize = self.checkBox_tokenization.isChecked()
            self.results = indexer.index(DOCS_FILE, self.Inverse, self.Tokenize, self.PorterStemmer)
            self.display_search_results(self.results)
        

    def search(self):
            """
            Perform search based on selected options.
            """
            self.Inverse = self.radioButton_inverse.isChecked()
            self.PorterStemmer = self.checkBox_porter_stemmer.isChecked()
            self.Tokenize = self.checkBox_tokenization.isChecked()
            query = self.search_bar.text().strip().lower()

            if self.checkBox_queries_dataset.isChecked():
                self.queries = loader.load_queries(QUERIES_FILE)
                self.judgements = loader.load_judgements(JUDGEMENTS_FILE)
                query_id = self.spinBox.value()
                query = self.queries[query_id-1][1]
                self.search_bar.setText(query)

            if self.radioButton_vsm.isChecked():
                self.sp = self.radioButton_SP.isChecked()
                self.cm = self.radioButton_CM.isChecked()
                self.jm = self.radioButton_JM.isChecked()
                self.results = vectorialModel.vectorial_model(query, self.Tokenize, self.PorterStemmer, self.sp, self.cm, self.jm)

            else :
                if self.radioButton_proba.isChecked():
                    self.K = self.lineEdit_K.text()
                    self.B = self.lineEdit_B.text()
                    self.results = probabilisticModel.probabilistic_model(query, self.Tokenize, self.PorterStemmer, float(self.K), float(self.B))

                elif self.radioButton_bool.isChecked():
                    self.results = booleanModel.boolean_model(query, self.Tokenize, self.PorterStemmer)

            if self.checkBox_queries_dataset.isChecked():
                    relevant_docs = self.judgements[query_id]

                    if self.radioButton_bool.isChecked():
                        if self.results is not None:
                            self.results = list(self.results.items())
                        else:
                            metric = None
                            plot = None
                    else:
                        selected_docs = [item[0] for item in self.results]
                        selected_relevant_docs = [doc for doc in relevant_docs if doc in selected_docs]

                        metric = metrics.get_metrics(selected_docs, selected_relevant_docs, relevant_docs)
                        plot = metrics.plot_precision_recall_curve(selected_docs, relevant_docs)

                    self.display_search_results2(self.results, metric, plot)
            else:
                self.display_search_results2(self.results)

    
    def display_search_results(self, results):
        """
        Display search results in the table widget.
        """
        self.tableWidget.setHorizontalHeaderLabels(["Key", "Value", "Frequency", "Weight"])
        self.tableWidget.setRowCount(0)
        
        for key1, values in results.items():
            for value in values:
                rowPosition = self.tableWidget.rowCount()
                self.tableWidget.insertRow(rowPosition)
                self.tableWidget.setItem(rowPosition, 0, QTableWidgetItem(str(key1)))
                self.tableWidget.setItem(rowPosition, 1, QTableWidgetItem(str(value[0])))
                self.tableWidget.setItem(rowPosition, 2, QTableWidgetItem(str(value[1])))
                self.tableWidget.setItem(rowPosition, 3, QTableWidgetItem(f"{value[2]:.4f}"))
                
        total_width = self.tableWidget.viewport().width()
        column_width = int(total_width / self.tableWidget.columnCount())
        for column in range(self.tableWidget.columnCount()):
            self.tableWidget.setColumnWidth(column, column_width)
    
    def display_search_results2(self, results, metric=None, plot=None):
        self.tableWidget.setHorizontalHeaderLabels(["N Doc", "Relevance"])
        self.tableWidget.setRowCount(0)

        if self.radioButton_bool.isChecked():
            if self.results != None:
                self.results = list(self.results.items())

        if self.radioButton_bool.isChecked() and self.results == None:
                rowPosition = self.tableWidget.rowCount()
                self.tableWidget.insertRow(rowPosition)
                self.tableWidget.setItem(rowPosition, 0, QTableWidgetItem('Invalid query !'))
        
        else:  
            for (key, value) in self.results:
                    rowPosition = self.tableWidget.rowCount()
                    self.tableWidget.insertRow(rowPosition)
                    self.tableWidget.setItem(rowPosition, 0, QTableWidgetItem(str(key)))
                    if self.radioButton_bool.isChecked():
                        self.tableWidget.setItem(rowPosition, 1, QTableWidgetItem(str(value)))
                    else:
                        self.tableWidget.setItem(rowPosition, 1, QTableWidgetItem(f"{float(value):.4f}"))

        if self.results is not None and self.checkBox_queries_dataset.isChecked():
            total_width = self.tableWidget.viewport().width()
            column_width = int(total_width / self.tableWidget.columnCount())
            for column in range(self.tableWidget.columnCount()):
                self.tableWidget.setColumnWidth(column, column_width)

            precision_value, precision_5, precision_10, recall_value, f_score_value = metric
            self.label_precision.setText(f"Precision = {precision_value:.4f}")
            self.label_p5.setText(f"P@5: {precision_5:.4f}")
            self.label_p10.setText(f"P@10: {precision_10:.4f}")
            self.label_recall.setText(f"Recall: {recall_value:.4f}")
            self.label_fscore.setText(f"F-score: {f_score_value:.4f}")

            # Set the plot
            pixmap = QPixmap('plots\precision_recall_curve.png')
            item = QGraphicsPixmapItem(pixmap)
            scene = QGraphicsScene()
            scene.addItem(item)

            self.plot.setScene(scene)


def main():
    app = QApplication([])
    window = MyGUI()
    app.exec_()


if __name__ == '__main__':
    main()