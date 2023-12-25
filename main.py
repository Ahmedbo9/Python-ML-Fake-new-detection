import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# data pre-processing
# loading dataset to panda dataframe

news_dataset = pd.read_csv('train.csv')
# printing the first five rows

news_dataset.isnull().sum()
news_dataset.fillna('')
# merging author and title
news_dataset['content'] = news_dataset['author'] + '' + news_dataset['title']

port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))


def stemming(content):
    """
    Process the input text content by:
    - Removing non-alphabetic characters
    - Converting to lowercase
    - Splitting into words
    - Applying stemming
    - Removing stopwords
    - Rejoining into a processed string
    """
    if not isinstance(content, str):
        return ""

    # Remove non-alphabetic characters and convert to lowercase
    processed_content = re.sub('[^a-zA-Z]', ' ', content).lower()

    # Split into words, apply stemming and remove stopwords
    stemmed_content = ' '.join([port_stem.stem(word) for word in processed_content.split() if word not in stop_words])

    return stemmed_content


news_dataset['content'] = news_dataset['content'].astype(str)  # Ensure all entries are strings
news_dataset['content'] = news_dataset['content'].apply(stemming)

# separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values


"""
   convert textual data to vectorized data
   
"""
vectorizer = TfidfVectorizer() #count the number of word in texte => give it a numerical value
vectorizer.fit(X) #apply to content
X = vectorizer.transform(X) #convert value to features

"""
Split data into training and testing sets
"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,stratify=Y, random_state=2)

"""
Training logistic regression model for binary classification

Y= 1 / 1+e^-z and  X = w.X+b
"""
model = LogisticRegression()
model.fit(X_train, Y_train)
"""
accuracy score on training set
"""
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(train_predictions, Y_train)

print('accuracy for train data: ', train_accuracy) #0.979 = good

"""
accuracy score on testing set
"""
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(test_predictions, Y_test)

print('accuracy for test data: ', test_accuracy) #0.961 = good

"""
building a predictive system 
"""
X_new = X_test[15]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')

print(Y_test[3])

