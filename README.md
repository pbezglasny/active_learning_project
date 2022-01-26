<!-- 
Instructions: 
- The report (report.md/report.ipynb ) should be in the root of your repository of a project
- The link to the repository have to be shared with us 
- Weekly report can be built in md-file or ipynb file 
- All reports for each week should be written into one file 
- Each week should be in a separated section in the file, see as shown in this file 
- The report should contain subsections TODO / WIP (work in progress) / Done / Issues 
- Each section should contain a list of works and their descriptions 
- Adding pictures / graphs / code inserts to md / ipynb cells can improve your report 
- The deadline is 11.59 pm UTC -12h (anywhere on earth)
 -->
 
# Project report are located in report.md file

# Week 1

TODO:
 - Idea to implement pipeline of training, which will take dialog with the worst
performance from previous epoch.
 - ...
 
WIP:
 - Added train method, which take data sampler to get dialog based on strategy
(get dialog with bad performance or get random dialogs)
   - Added WorstDialogSampler, which select dialogs with bad performance
   - Added classes for calculating performance of dialogs
 - ... 

Done:
 - I have pipeline for training model based on chosen strategy.
 
# Week 2

TODO:
 - Update train pipeline: now model train only on part of data. At each epoch dialog with 
bad performance are chosen to train at epoch
 
WIP:
 - Added trainer class, which encapsulate train steps.
 - Added sampler, which remove used dialogs after epoch
 - Added metric classes

Done:
 - Added classes that implement logic from point above. 

# Week 3

TODO:
 - Train different models with different parameter(number of epoch, percent of choosing 
bad dialogs)
 
WIP:
 - Updated run train script to use different models.

Done:
 - Trained 4 models(bert-base-uncased, distilbert-base-uncased, 
bert-base-multilingual-cased, bert-base-cased)
with different transformer underneath.