from datetime import date
from datetime import datetime

from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from fpdf import FPDF


# for generate pdf report
class PDF(FPDF):
    def __init__(self):
        super().__init__()

    def header(self):
        self.set_font('Arial', '', 12)
        # self.cell(0, 8, 'Decision Tree Implementation Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 12)
        self.cell(0, 8, f'Page {self.page_no()}', 0, 0, 'C')


# To calc the accuracy
def measureAccuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)


# Experiment with specific train_test by Decision Tree Algorithm
def Experiment_Utility(X, Y, splitRatio):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=splitRatio)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    return measureAccuracy(predictions, y_test), clf.tree_.node_count


# To Return Statistics (Mean, Max and Min)
def GetStats(array):
    return np.array([np.mean(array), np.max(array), np.min(array)])


# To Return the sizes and accuracies of these trees in each experiment (learning rate => 25%)
def Experiment(X, Y, splitRatio):
    accur = np.zeros(5)
    treeSize = np.zeros(5)
    for i in range(0, 5):
        accur[i], treeSize[i] = Experiment_Utility(X, Y, splitRatio)
    return accur, treeSize


# plotting against Y Axis (Accuracy, TreeSize)
def plotting(y_axis, fileName):
    set_size = [30, 40, 50, 60, 70]
    plt.plot(set_size, y_axis, color='red')
    plt.xlabel('set_size')
    plt.ylabel('accuracy')
    plt.savefig(fileName + ".png")
    plt.show()


def main():
    # Read the data
    data = pd.read_csv("BankNote_Authentication.csv")

    # Separate X and Y inside the data
    X = data.drop(columns=['class']).to_numpy().reshape(-1, 4)
    Y = data['class']

    # Run with 0.25 training ratio
    exp1 = pd.DataFrame(Experiment(X, Y, 0.75))

    # Initialize the accuracy and size Matrices
    accuracy = [[0] * 3] * 5
    size = [[0] * 3] * 5

    # Run with [0.3,0.4,0.5,0.6,0.7] training ratios
    for i in range(3, 8):
        accur, treeSize = Experiment(X, Y, (10 - i) / 10)
        accuracy[i - 3] = GetStats(accur)
        size[i - 3] = GetStats(treeSize)

    # Extracting Means and plotting them
    acc_means = ([row[0] for row in accuracy])
    plotting(acc_means, "exp1")
    size_means = ([row[0] for row in size])
    plotting(size_means, "exp2")

    # Generating Pdf
    margin = 8
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(190, 0, 0)
    pdf.cell(w=0, h=20, txt="Decision Tree Classifier Report", ln=1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(w=30, h=margin, txt="Date: ", ln=0)
    pdf.cell(w=30, h=margin, txt=str(date.today().strftime("%d/%m/%Y")), ln=1)
    pdf.cell(w=30, h=margin, txt="Time: ", ln=0)
    pdf.cell(w=30, h=margin, txt=str(datetime.now().strftime("%H:%M:%S")), ln=1)
    pdf.cell(w=30, h=margin, txt="Authors: ", ln=0)
    pdf.cell(w=30, h=margin, txt="Khaled Ashraf, Noura Ashraf, Samaa Khalifa", ln=1)
    pdf.ln(14)

    # First Experiment
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(190, 0, 0)
    pdf.cell(0, 8, 'Experiment 1', 0, 10, 'C')
    pdf.ln(margin)

    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(16, 63, 145)
    # Table Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(w=95, h=margin, txt='Set Size', border=1, ln=0, align='C', fill=True)
    pdf.cell(w=95, h=margin, txt='Accuracy', border=1, ln=1, align='C', fill=True)
    # Table contents
    pdf.set_font('Arial', '', 16)
    pdf.set_text_color(0, 0, 0)

    for (index, data) in exp1.iteritems():
        pdf.cell(w=95, h=margin,
                 txt=str(data.values[1]),
                 border=1, ln=0, align='C')
        pdf.cell(w=95, h=margin,
                 txt=str(data.values[0]),
                 border=1, ln=1, align='C')
    pdf.ln(15)

    # Second Experiment
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(190, 0, 0)
    pdf.cell(0, 8, 'Experiment 2', 0, 10, 'C')
    pdf.ln(margin)

    # First table
    pdf.set_text_color(16, 63, 145)
    pdf.cell(0, 8, 'Accuracy for each iteration', 0, 10, 'L')
    pdf.ln(margin)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(16, 63, 145)
    # Table Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(w=30, h=margin, txt='Iteration', border=1, ln=0, align='C', fill=True)
    pdf.cell(w=52, h=margin, txt='Mean', border=1, ln=0, align='C', fill=True)
    pdf.cell(w=52, h=margin, txt='Max', border=1, ln=0, align='C', fill=True)
    pdf.cell(w=52, h=margin, txt='Min', border=1, ln=1, align='C', fill=True)
    # Table contents
    pdf.set_font('Arial', '', 16)
    pdf.set_text_color(0, 0, 0)
    index = 30
    for item in accuracy:
        pdf.cell(w=30, h=margin,
                 txt=str(index) + "%",
                 border=1, ln=0, align='C')
        pdf.cell(w=52, h=margin,
                 txt=str(round(item[0], 5)),
                 border=1, ln=0, align='C')
        pdf.cell(w=52, h=margin,
                 txt=str(round(item[1], 5)),
                 border=1, ln=0, align='C')
        pdf.cell(w=52, h=margin,
                 txt=str(round(item[2], 5)),
                 border=1, ln=1, align='C')
        index += 10

    # Second Table
    pdf.ln(50)
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(16, 63, 145)
    pdf.cell(0, 8, 'Size for each iteration', 0, 10, 'L')
    pdf.ln(margin)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(16, 63, 145)
    # Table Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(w=30, h=margin, txt='Iteration', border=1, ln=0, align='C', fill=True)
    pdf.cell(w=52, h=margin, txt='Mean', border=1, ln=0, align='C', fill=True)
    pdf.cell(w=52, h=margin, txt='Max', border=1, ln=0, align='C', fill=True)
    pdf.cell(w=52, h=margin, txt='Min', border=1, ln=1, align='C', fill=True)
    # Table contents
    pdf.set_font('Arial', '', 16)
    pdf.set_text_color(0, 0, 0)
    index = 30
    for item in size:
        pdf.cell(w=30, h=margin,
                 txt=str(index) + "%",
                 border=1, ln=0, align='C')

        pdf.cell(w=52, h=margin,
                 txt=str(item[0]),
                 border=1, ln=0, align='C')
        pdf.cell(w=52, h=margin,
                 txt=str(item[1]),
                 border=1, ln=0, align='C')
        pdf.cell(w=52, h=margin,
                 txt=str(item[2]),
                 border=1, ln=1, align='C')
        index += 10

    pdf.ln(15)
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(16, 63, 145)
    pdf.cell(0, 8, 'Plotting accuracy against training set size', 0, 10, 'C')
    pdf.image('./exp1.png', x=50, y=None, w=110, h=110, type='PNG', link='')
    pdf.ln(80)
    pdf.set_font('Arial', 'B', 18)
    pdf.set_text_color(16, 63, 145)
    pdf.cell(0, 8, 'Plotting the final tree size against training set size', 0, 10, 'C')
    pdf.image('./exp2.png', x=50, y=None, w=110, h=110, type='PNG', link='')
    pdf.ln(margin)
    pdf.output(f'./DecisionTreeReport.pdf', 'F')


if __name__ == "__main__":
    main()
