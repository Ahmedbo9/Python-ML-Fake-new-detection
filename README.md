# Fake News Detection Project

## Overview

This project demonstrates a machine learning solution for detecting fake news, implemented using Python. It comprises a machine learning model for classification, an API developed with FastAPI, and a user-friendly web interface created using Flask.



### DataSet

DataSet source : https://www.kaggle.com/c/fake-news/data?select=train.csv



### Machine Learning Model

The core of this project is a fake news detection model built using Python's Scikit-Learn library. The model is trained on a dataset that includes various news articles, labeled as either 'real' or 'fake'. The preprocessing of the text data involves cleaning, stemming, and vectorization (TF-IDF). A logistic regression classifier is then trained on this processed data.

**Key Technologies:**
- Pandas for data manipulation.
- NLTK for text processing.
- Scikit-Learn for building and evaluating the logistic regression model.

### API with FastAPI

A FastAPI server exposes the machine learning model as an API endpoint. This allows for easy integration with web or mobile applications. The API accepts a piece of news text as input and returns a prediction indicating whether the news is real or fake.

**Key Features:**
- Fast, asynchronous API handling.
- Endpoint for processing and predicting news text.

### Flask Web Application

A Flask web application serves as the frontend for this project. Users can input news text into a form, which is then sent to the FastAPI backend for prediction. The results are displayed on the web page, providing an intuitive interface for interacting with the machine learning model.

**Key Features:**
- Integration with the FastAPI backend for real-time predictions.

### Setup and Installation

### Prerequisites

- Python 3.8 or above
- pip (Python package manager)

### Installation Steps
`git clone repository link`

`pip install -r requirements.txt`

`uvicorn main:app --reload --port 8000`

`python flask_app.py`









