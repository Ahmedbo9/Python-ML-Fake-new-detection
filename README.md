# News Classification Project

This project aims to classify news as real or fake using logistic regression. It involves pre-processing the textual data, converting it to vectorized form, and then applying a logistic regression model for classification.

## Prerequisites

- Python 3
- Libraries: pandas, NLTK, scikit-learn

## Installation

1. Clone the repository or download the source code.
2. Install required dependencies:
3. Download NLTK stopwords (if not already downloaded):


## Dataset

DataSet source :  https://www.kaggle.com/c/fake-news/data?select=train.csv

## Usage

1. Load  `train.csv` file.
2. Run the Python script to train the model and predict news authenticity.

## Features

- Data pre-processing includes:
- Handling missing values.
- Merging author and title.
- Text cleaning and stemming.
- Textual data is converted to numerical data using TF-IDF Vectorization.
- The dataset is split into training and testing sets.
- A logistic regression model is trained for binary classification.
- The model's accuracy is evaluated on both training and testing sets.

## Output

The script will output the accuracy scores for both training and testing data. Additionally, it will classify a sample news item as real or fake.



