#Abstract

----
In this project was develop train pipeline of active learning for classification of 
dialog acts. After each epoch some dialogs with bad performance were chosen to continue 
train model  

#Approach

---
Top level
---
---
Three transformer based NLP models(bert-base-uncased, distilbert-base-uncased, 
bert-base-cased) were chosen to train. 
For first epoch 10% of train data were selected to initial train the model. Then at
each epoch some percents of data with the worst performance were used to continue train models.


Implementation notes
---
---


#Results

---

#Conclusion

---

 