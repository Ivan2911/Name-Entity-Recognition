# Name-Entity-Recognition

=======
This project is a part of the NLP Project from University of Alabama at Birmingham
=======

## Project Description
This project focuses on applying pretrained language representation models such as BERT and DistilBERT to perform classification on the Groningen Meaning Bank dataset. The model developed predicts the IOB tag for tokens in each sentence. We have achieved results of around 98% accuracy for for models and and F1-SCORE(macro) of 69% and 75% for BERT and DistilBERT respectively.

### Methods Used
* Machine Learning
* Deep Learning
* Data Visualization
* Classification

### Technologies
* Python
* Scikit-learn
* NumPy
* Pandas, jupyter
* PyTorch
* Keras

# Requirements
- [Python3](https://www.python.org)
- [Scikit-Learn](https://scikit-learn.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Transformers](https://huggingface.co/docs/transformers/index)(Hugging Face)
- [Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [PyTorch](https://pytorch.org/)
- [Jupyter](https://jupyter.org/)

# Getting Started
- Clone this repo
- [Download the Dataset](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus?select=ner_dataset.csv)
- Install the requirements.txt
- Run the jupyter notebooks (bert_best or distilbert)