# Banknote Authentication Decision Tree

This Python code utilizes the decision tree algorithm from the scikit-learn library to perform banknote authentication. The code aims to analyze the impact of different train-test split ratios and training set sizes on the accuracy and size of the learned decision tree.

## Dataset

The code uses the "BankNote_Authentication.csv" dataset, which contains four features (variance, skew, curtosis, and entropy) and a class attribute indicating whether a banknote is real or forged.

## Requirements

The following libraries are imported in the code:

- `sklearn.tree`: Provides the decision tree classifier.
- `pandas`: Used for data manipulation and analysis.
- `sklearn.model_selection.train_test_split`: Splits the data into training and testing sets.
- `numpy`: Handles mathematical operations and array manipulation.
- `matplotlib.pyplot`: Enables data visualization.

## Functions

### `measureAccuracy(y_pred, y_test)`

Calculates the accuracy of the predicted labels (`y_pred`) compared to the actual labels (`y_test`). Returns the accuracy as a floating-point value.

### `Experiment_Utility(X, Y, splitRatio)`

Performs an experiment with a specific train-test split ratio (`splitRatio`) using the decision tree algorithm. Splits the data into training and testing sets, fits the decision tree model, and predicts the labels for the testing set. Returns the accuracy and the number of nodes in the decision tree.

### `GetStats(array)`

Calculates the mean, maximum, and minimum values of an input array. Returns the statistics as a NumPy array.

### `Experiment(X, Y, splitRatio)`

Performs multiple experiments with a fixed train-test split ratio (`splitRatio`). Reruns the experiment five times with different random splits of the data. Returns the accuracies and tree sizes for each experiment.

### `plotting(y_axis, fileName)`

Plots the y-axis values against the training set size. Saves the plot as an image file with the specified `fileName`.

### `main()`

The main function reads the dataset, separates the features (X) and the labels (Y), and initializes matrices for accuracy and tree size statistics. It then runs two sets of experiments:

### Experiment 1: Fixed train-test split ratio
- The function runs the experiment with a 75% training ratio, recording the accuracies and tree sizes for each iteration.
- The size of each iteration is displayed in the following table:

<div class="WordSection1" align="center"><table class="MsoNormalTable" border="1" cellspacing="0" cellpadding="0" style="margin-left:5.8pt;border-collapse:collapse;mso-table-layout-alt:fixed;
 border:none;mso-border-alt:solid black .75pt;mso-yfti-tbllook:480;mso-padding-alt:
 0cm 0cm 0cm 0cm;mso-border-insideh:.75pt solid black;mso-border-insidev:.75pt solid black">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes;height:21.9pt">
  <td width="359" valign="top" style="width:269.3pt;border:solid black 1.0pt;
  mso-border-alt:solid black .75pt;background:#103E91;padding:0cm 0cm 0cm 0cm;
  height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:55.1pt;
  margin-bottom:0cm;margin-left:55.65pt;margin-bottom:.0001pt"><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-hansi-font-family:&quot;Arial MT&quot;;
  mso-bidi-font-family:&quot;Arial MT&quot;;color:white">Set Size</span></b><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-hansi-font-family:&quot;Arial MT&quot;;
  mso-bidi-font-family:&quot;Arial MT&quot;"><o:p></o:p></span></b></p>
  </td>
  <td width="359" valign="top" style="width:269.3pt;border:solid black 1.0pt;
  border-left:none;mso-border-left-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  background:#103E91;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:55.1pt;
  margin-bottom:0cm;margin-left:55.65pt;margin-bottom:.0001pt"><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-hansi-font-family:&quot;Arial MT&quot;;
  mso-bidi-font-family:&quot;Arial MT&quot;;color:white">Accuracy</span></b><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-hansi-font-family:&quot;Arial MT&quot;;
  mso-bidi-font-family:&quot;Arial MT&quot;"><o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1;height:21.9pt">
  <td width="359" valign="top" style="width:269.3pt;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:55.1pt;
  margin-bottom:0cm;margin-left:55.65pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">25.0<o:p></o:p></span></p>
  </td>
  <td width="359" valign="top" style="width:269.3pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:55.1pt;
  margin-bottom:0cm;margin-left:55.65pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">0.9620991253644315<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2;height:21.9pt">
  <td width="359" valign="top" style="width:269.3pt;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:55.1pt;
  margin-bottom:0cm;margin-left:55.65pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">31.0<o:p></o:p></span></p>
  </td>
  <td width="359" valign="top" style="width:269.3pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:55.1pt;
  margin-bottom:0cm;margin-left:55.65pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">0.9630709426627794<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3;height:21.9pt">
  <td width="359" valign="top" style="width:269.3pt;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:55.1pt;
  margin-bottom:0cm;margin-left:55.65pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">39.0<o:p></o:p></span></p>
  </td>
  <td width="359" valign="top" style="width:269.3pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:55.1pt;
  margin-bottom:0cm;margin-left:55.65pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">0.956268221574344<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4;height:21.9pt">
  <td width="359" valign="top" style="width:269.3pt;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:55.1pt;
  margin-bottom:0cm;margin-left:55.65pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">27.0<o:p></o:p></span></p>
  </td>
  <td width="359" valign="top" style="width:269.3pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:55.1pt;
  margin-bottom:0cm;margin-left:55.65pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">0.967930029154519<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5;mso-yfti-lastrow:yes;height:21.9pt">
  <td width="359" valign="top" style="width:269.3pt;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:55.1pt;
  margin-bottom:0cm;margin-left:55.65pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">31.0<o:p></o:p></span></p>
  </td>
  <td width="359" valign="top" style="width:269.3pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:55.1pt;
  margin-bottom:0cm;margin-left:55.65pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">0.9689018464528668<o:p></o:p></span></p>
  </td>
 </tr>
</tbody></table></div>

<hr>

### Experiment 2: Range of train-test split ratios
- The function iterates over a range of training set sizes (30% to 70%) and performs the experiment five times with different random seeds.
- For each training set size, it calculates the mean, maximum, and minimum accuracy and tree size for all iterations.
- The accuracy and tree size for each iteration are displayed in the following tables:


### Accuracy for each iteration ###
<div class="WordSection1" align="center"><table class="MsoNormalTable" border="1" cellspacing="0" cellpadding="0" style="margin-left:5.8pt;border-collapse:collapse;mso-table-layout-alt:fixed;
 border:none;mso-border-alt:solid black .75pt;mso-yfti-tbllook:480;mso-padding-alt:
 0cm 0cm 0cm 0cm;mso-border-insideh:.75pt solid black;mso-border-insidev:.75pt solid black">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes;height:21.9pt">
  <td width="113" valign="top" style="width:3.0cm;border:solid black 1.0pt;
  mso-border-alt:solid black .75pt;background:#103E91;padding:0cm 0cm 0cm 0cm;
  height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:9.3pt;
  margin-bottom:0cm;margin-left:9.85pt;margin-bottom:.0001pt"><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-hansi-font-family:&quot;Arial MT&quot;;
  mso-bidi-font-family:&quot;Arial MT&quot;;color:white">Iteration</span></b><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-hansi-font-family:&quot;Arial MT&quot;;
  mso-bidi-font-family:&quot;Arial MT&quot;"><o:p></o:p></span></b></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border:solid black 1.0pt;
  border-left:none;mso-border-left-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  background:#103E91;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-left:43.6pt"><b style="mso-bidi-font-weight:
  normal"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt;font-family:
  &quot;Arial&quot;,sans-serif;mso-hansi-font-family:&quot;Arial MT&quot;;mso-bidi-font-family:
  &quot;Arial MT&quot;;color:white">Mean</span></b><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt;font-family:&quot;Arial&quot;,sans-serif;
  mso-hansi-font-family:&quot;Arial MT&quot;;mso-bidi-font-family:&quot;Arial MT&quot;"><o:p></o:p></span></b></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border:solid black 1.0pt;
  border-left:none;mso-border-left-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  background:#103E91;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt;font-family:&quot;Arial&quot;,sans-serif;
  mso-hansi-font-family:&quot;Arial MT&quot;;mso-bidi-font-family:&quot;Arial MT&quot;;color:white">Max</span></b><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-hansi-font-family:&quot;Arial MT&quot;;
  mso-bidi-font-family:&quot;Arial MT&quot;"><o:p></o:p></span></b></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border:solid black 1.0pt;
  border-left:none;mso-border-left-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  background:#103E91;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt;font-family:&quot;Arial&quot;,sans-serif;
  mso-hansi-font-family:&quot;Arial MT&quot;;mso-bidi-font-family:&quot;Arial MT&quot;;color:white">Min</span></b><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-hansi-font-family:&quot;Arial MT&quot;;
  mso-bidi-font-family:&quot;Arial MT&quot;"><o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1;height:21.9pt">
  <td width="113" valign="top" style="width:3.0cm;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:9.3pt;
  margin-bottom:0cm;margin-left:9.85pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">30%<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.96774<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.97815<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.95421<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2;height:21.9pt">
  <td width="113" valign="top" style="width:3.0cm;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:9.3pt;
  margin-bottom:0cm;margin-left:9.85pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">40%<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.97282<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.97937<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.96723<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3;height:21.9pt">
  <td width="113" valign="top" style="width:3.0cm;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:9.3pt;
  margin-bottom:0cm;margin-left:9.85pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">50%<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.97376<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.98834<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.96064<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4;height:21.9pt">
  <td width="113" valign="top" style="width:3.0cm;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:9.3pt;
  margin-bottom:0cm;margin-left:9.85pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">60%<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.98069<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.98361<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.96903<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5;mso-yfti-lastrow:yes;height:21.9pt">
  <td width="113" valign="top" style="width:3.0cm;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:9.3pt;
  margin-bottom:0cm;margin-left:9.85pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">70%<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.97961<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.99029<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">0.9733<o:p></o:p></span></p>
  </td></tr></tbody></table></div>

### Size for each iteration ###
<div class="WordSection1" align="center"><div align="center"><table class="MsoNormalTable" border="1" cellspacing="0" cellpadding="0" style="border-collapse:collapse;mso-table-layout-alt:fixed;border:none;
 mso-border-alt:solid black .75pt;mso-yfti-tbllook:480;mso-padding-alt:0cm 0cm 0cm 0cm;
 mso-border-insideh:.75pt solid black;mso-border-insidev:.75pt solid black">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes;height:21.9pt">
  <td width="113" valign="top" style="width:3.0cm;border:solid black 1.0pt;
  mso-border-alt:solid black .75pt;background:#103E91;padding:0cm 0cm 0cm 0cm;
  height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:9.3pt;
  margin-bottom:0cm;margin-left:9.85pt;margin-bottom:.0001pt"><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-hansi-font-family:&quot;Arial MT&quot;;
  mso-bidi-font-family:&quot;Arial MT&quot;;color:white">Iteration</span></b><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-hansi-font-family:&quot;Arial MT&quot;;
  mso-bidi-font-family:&quot;Arial MT&quot;"><o:p></o:p></span></b></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border:solid black 1.0pt;
  border-left:none;mso-border-left-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  background:#103E91;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-left:43.6pt"><b style="mso-bidi-font-weight:
  normal"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt;font-family:
  &quot;Arial&quot;,sans-serif;mso-hansi-font-family:&quot;Arial MT&quot;;mso-bidi-font-family:
  &quot;Arial MT&quot;;color:white">Mean</span></b><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt;font-family:&quot;Arial&quot;,sans-serif;
  mso-hansi-font-family:&quot;Arial MT&quot;;mso-bidi-font-family:&quot;Arial MT&quot;"><o:p></o:p></span></b></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border:solid black 1.0pt;
  border-left:none;mso-border-left-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  background:#103E91;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:57.35pt;
  margin-bottom:0cm;margin-left:0cm;margin-bottom:.0001pt"><b style="mso-bidi-font-weight:
  normal"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt;font-family:
  &quot;Arial&quot;,sans-serif;mso-hansi-font-family:&quot;Arial MT&quot;;mso-bidi-font-family:
  &quot;Arial MT&quot;;color:white">Max</span></b><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt;font-family:&quot;Arial&quot;,sans-serif;
  mso-hansi-font-family:&quot;Arial MT&quot;;mso-bidi-font-family:&quot;Arial MT&quot;"><o:p></o:p></span></b></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border:solid black 1.0pt;
  border-left:none;mso-border-left-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  background:#103E91;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt;font-family:&quot;Arial&quot;,sans-serif;
  mso-hansi-font-family:&quot;Arial MT&quot;;mso-bidi-font-family:&quot;Arial MT&quot;;color:white">Min</span></b><b style="mso-bidi-font-weight:normal"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt;font-family:&quot;Arial&quot;,sans-serif;mso-hansi-font-family:&quot;Arial MT&quot;;
  mso-bidi-font-family:&quot;Arial MT&quot;"><o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1;height:21.9pt">
  <td width="113" valign="top" style="width:3.0cm;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:9.3pt;
  margin-bottom:0cm;margin-left:9.85pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">30%<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">31.8<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:57.35pt;
  margin-bottom:0cm;margin-left:0cm;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">37.0<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">25.0<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2;height:21.9pt">
  <td width="113" valign="top" style="width:3.0cm;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:9.3pt;
  margin-bottom:0cm;margin-left:9.85pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">40%<o:p></o:p></span></p>
  </td><td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">37.4<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:57.35pt;
  margin-bottom:0cm;margin-left:0cm;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">41.0<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">35.0<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3;height:21.9pt">
  <td width="113" valign="top" style="width:3.0cm;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:9.3pt;
  margin-bottom:0cm;margin-left:9.85pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">50%<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">35.8<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:57.35pt;
  margin-bottom:0cm;margin-left:0cm;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">45.0<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">27.0<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4;height:21.9pt">
  <td width="113" valign="top" style="width:3.0cm;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:9.3pt;
  margin-bottom:0cm;margin-left:9.85pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">60%<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">41.0<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:57.35pt;
  margin-bottom:0cm;margin-left:0cm;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">47.0<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">35.0<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5;mso-yfti-lastrow:yes;height:21.9pt">
  <td width="113" valign="top" style="width:3.0cm;border:solid black 1.0pt;
  border-top:none;mso-border-top-alt:solid black .75pt;mso-border-alt:solid black .75pt;
  padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:9.3pt;
  margin-bottom:0cm;margin-left:9.85pt;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">70%<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">47.0<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph" style="margin-top:.75pt;margin-right:57.35pt;
  margin-bottom:0cm;margin-left:0cm;margin-bottom:.0001pt"><span style="font-size:16.0pt;mso-bidi-font-size:11.0pt">51.0<o:p></o:p></span></p>
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
  none;border-bottom:solid black 1.0pt;border-right:solid black 1.0pt;
  mso-border-top-alt:solid black .75pt;mso-border-left-alt:solid black .75pt;
  mso-border-alt:solid black .75pt;padding:0cm 0cm 0cm 0cm;height:21.9pt">
  <p class="TableParagraph"><span style="font-size:16.0pt;mso-bidi-font-size:
  11.0pt">41.0<o:p></o:p></span></p></td></tr></tbody></table></div><br></div>
  
     
## Usage

To run the code, follow these steps:

1. Install the required libraries: `sklearn`, `pandas`, `numpy`, and `matplotlib.pyplot`.
2. Download the "BankNote_Authentication.csv" dataset and place it in the same directory as the code file.
3. Run the code. The main function will execute the experiments and generate the accuracy and tree size results.
4. The code will also generate plots showing the accuracy and tree size against the training set size.


## Conclusion

In conclusion, this Python code provides a practical implementation of banknote authentication using a decision tree algorithm. It allows for experimentation with different train-test split ratios and training set sizes, providing insights into how these factors affect the accuracy and size of the decision tree model.


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.


## Team

- [Khaled Ashraf Hanafy Mahmoud - 20190186](https://github.com/KhaledAshrafH).
- [Noura Ashraf Abdelnaby Mansour - 20190592](https://github.com/NouraAshraff).
- [Samaa Khalifa Elsayed Othman - 20190247](https://github.com/SamaaKhalifa).

## License

This program is licensed under the [MIT License](LICENSE.md).
