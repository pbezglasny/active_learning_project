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
 