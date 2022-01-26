# Abstract

----
In this project was develop train pipeline of active learning for classification of dialog acts. After each epoch some
dialogs with bad performance were chosen to continue train model

# Approach

---
Top level
---
---
Three transformer based NLP models(bert-base-uncased, distilbert-base-uncased,
bert-base-cased) were chosen to train.
For first epoch 10% of train data were selected to initial train the model. 
Then at each epoch some percents of data
with the worst performance were used to continue train models.  
The trained model will be compared with model, which train same size of 
random dialog data.

Implementation notes
---
---
To load models and dataset huggingface library was used.  
Additionally, scripts were added:  
* trainer.py - script contains Trainer class, which introduced steps of 
training/evaluation of pipeline.
* metric.py - custom classes to evaluate model performance to
select dialogs with bad performance.
* data.py - script contains pytorch data samplers to resample data to select 
dialogs with bad performance or select random dialogs.


# Results

---

# Conclusion

---

 