# Decision-Tree
Decision Tree using Scikit-learn and apply it on dataset "BankNote_Authentication.csv" to tell whether a banknote is real or not.

Whenever you go to the bank to deposit some cash money, the cashier places banknotes in a machine that tells whether a banknote is real or not. In the “BankNote_Authentication.csv” we have four features: variance, skew, curtosis and entropy and the class attribute refers to whether or not the banknote is real or forged.

# Experiment 1
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

# Experiment 2

## Accuracy for each iteration ##
<div class="WordSection1" align="center">
<table class="MsoNormalTable" border="1" cellspacing="0" cellpadding="0" style="margin-left:5.8pt;border-collapse:collapse;mso-table-layout-alt:fixed;
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
  </td>
 </tr>
</tbody></table>
</div>

## Size for each iteration ##
<div class="WordSection1" align="center">
<div align="center">

<table class="MsoNormalTable" border="1" cellspacing="0" cellpadding="0" style="border-collapse:collapse;mso-table-layout-alt:fixed;border:none;
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
  </td>
  <td width="197" valign="top" style="width:147.4pt;border-top:none;border-left:
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
  11.0pt">41.0<o:p></o:p></span></p>
  </td>
 </tr>
</tbody></table>

</div><br>
</div>
