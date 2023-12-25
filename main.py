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
vectorizer = TfidfVectorizer()  # count the number of word in texte => give it a numerical value
vectorizer.fit(X)  # apply to content
X = vectorizer.transform(X)  # convert value to features

"""
Split data into training and testing sets
"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

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

print('accuracy for train data: ', train_accuracy)  # 0.979 = good

"""
accuracy score on testing set
"""
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(test_predictions, Y_test)

print('accuracy for test data: ', test_accuracy)  # 0.961 = good

"""
building a predictive system 

X_new = X_test[15]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')

print(Y_test[15])
"""

"""
testing with real data from : https://www.ctvnews.ca/health/coronavirus/what-we-know-so-far-about-the-new-covid-variant-including-symptoms-1.6697096
"""


def preprocess_and_predict(input_text):
    """
    Process the input text and predict using the trained model
    """
    # Preprocess the input text
    preprocessed_text = stemming(input_text)

    # Vectorize the preprocessed text
    vectorized_input = vectorizer.transform([preprocessed_text])

    # Predict using the trained model
    prediction = model.predict(vectorized_input)

    # Print the result
    if prediction[0] == 0:
        return 'The news is Real'
    else:
        return 'The news is Fake'


# Example usage
real_news = ('What we know so far about the new COVID variant, including symptoms NOW PLAYING CTV Medical Expert Dr. '
             'Marla Shapiro discusses new vaccine recommendations and the new COVID variant BA.2.86. 03:16 Health '
             'Canada considers new variant vaccine UP NEXT Dr. Kashif Pirzada explains how the new COVID variant '
             'BA.2.86 is different from others and how Canadians can stay safe. 04:11 What we know about the new '
             'COVID variant  Dr. Isaac Bogoch discusses how early signs indicate a new wave of COVID infections '
             'might be coming to Canada. 04:00 COVID-19 hospitalizations increasing  Health experts say Ontario is '
             'in another COVID-19 wave in conjunction with the emergence of two new Omicron variants. 02:39 New '
             'COVID-19 wave fuelled by Omicron variant  Epidemiologist Dr. Christopher Labos discusses the different '
             'types of immunity and the importance of vaccinations. 04:52 Majority showing signs of COVID-19 '
             'immunity  Dr. Isaac Bogoch discusses the latest COVID-19 variant and what you need to know ahead of a '
             'possible rise in cases. 04:31 New COVID variant: Bogoch on what to know  Mitchell Consky CTVNews.ca '
             'Writer Follow | Contact Updated Dec. 21, 2023 12:23 p.m. EST Published Dec. 21, 2023 12:22 p.m. EST '
             'Share twitter reddit More share options As respiratory virus season kicks off in North America, '
             'a heavily mutated COVID-19 variant is expected to keep spreading throughout the holidays, but experts '
             'say the risk to public health remains “low.” The variant, called JN.1, is classified by the World '
             'Health Organization (WHO) as a “variant of interest,” but health experts say the variant does not show '
             'any signs of more severe disease if contracted.  “Considering the available, yet limited evidence, '
             'the additional public health risk posed by JN.1 is currently evaluated as low at the global level,'
             '” WHO said in a report(opens in a new tab) that evaluates the initial risk of the strain.  WHO '
             'anticipates that this variant “may cause an increase in SARS-CoV-2 cases amid asurge of infections of '
             'other viral and bacterial infections, especially in countries entering the winter season.” Its a '
             'similar message to that of the U.S.-based Centers for Disease Control (CDC), where experts said(opens '
             'in a new tab) last week that low vaccination rates compared to this time last year are leaving the '
             'public at a greater risk of serious illness. According to WHO, the countries that reported the largest '
             'proportion of JN.1 sequences submitted as of last week were France (20.1 per cent, or 1,'
             '552 sequences), the U.S. (14 per cent, 1,072 sequences), Singapore (12 per cent, 934 sequences), '
             'and Canada (6.8 per cent, 512 sequences).  WHO added that it is yet to be determined whether the high '
             'transmissibility of JN.1 could be associated with “primary human nasal epithelial cells,'
             '” which comprise the lining of the nasal passages in the human respiratory system, or whether this is '
             'linked to “non-spike proteins” in the strain, meaning differing functions of the variant the evade '
             'immune responses.  SYMPTOMS According to the CDC(opens in a new tab), symptoms of this strain are no '
             'different from previous variants of COVID-19, most specifically the Omicron BA.2 variant, '
             'of which this strain is a descendent.  “The types of symptoms and how severe they are usually depend '
             'more on a person’s immunity and overall health rather than which variant causes the infection,'
             '” the CDC said on its website.  Typical symptoms include a dry cough, headache, fever, and fatigue.  '
             'RELATED IMAGES  FILE - A Moderna Spikevax COVID-19 vaccine is seen at a drugstore in Cypress, Texas, '
             'Sept. 20, 2023. (Melissa Phillip/Houston Chronicle via AP, File) FILE - A Moderna Spikevax COVID-19 '
             'vaccine is seen at a drugstore in Cypress, Texas, Sept. 20, 2023. (Melissa Phillip/Houston Chronicle '
             'via AP, File)')

fake_news = ('Unsubstantiated Reports Claim New COVID-19 Variant "ZX-5" Grants Superhuman AbilitiesIn a surprising '
             'and scientifically unfounded turn of events, rumors are circulating about a new COVID-19 variant, '
             'dubbed "ZX-5", which allegedly bestows superhuman abilities upon those infected. Sources lacking '
             'credibility suggest that this variant, first identified in a remote village, has led to individuals '
             'exhibiting extraordinary strength, speed, and cognitive abilities.Dr. Alex Redfield, whose credentials '
             'and existence remain unverified, claims, The ZX-5 variant is unlike anything we ve seen. It appears to '
             'interact with human DNA in a way that enhances physical and mental faculties These assertions, however, '
             'are not supported by any published scientific research or peer-reviewed studies.ealth officials '
             'worldwide, including the World Health Organization (WHO) and the Centers for Disease Control and '
             'Prevention (CDC), have categorically denied these claims, emphasizing the lack of evidence and the '
             'dangers of spreading misinformation. "There is absolutely no scientific basis for these assertions. '
             'COVID-19 variants can be more contagious or severe, but they do not grant superpowers," stated a WHO '
             'spokesperson.Despite this, the story has gained traction on social media, fueled by sensationalist '
             'reporting and conspiracy theorists. In one video, a self-proclaimed expert demonstrates what he claims '
             'to be "enhanced reflexes" due to the ZX-5 variant, but this has been widely debunked as a simple magic '
             'trick.The alleged origin of ZX-5, a small village in an undisclosed location, has reportedly become a '
             'site of pilgrimage for those seeking these supposed abilities. Local authorities are struggling to '
             'manage the influx of visitors, many of whom are disregarding basic health and safety guidelines amid '
             'the ongoing pandemic.In Canada, Dr. Emily Norton, a virologist with no record in the scientific '
             'community, has been quoted as saying, "We are on the brink of a new era in human evolution, '
             'thanks to ZX-5." This statement has been heavily criticized by reputable scientists, who warn against '
             'the dangers of such unfounded claims.Symptoms of the so-called ZX-5 variant are said to include a '
             'temporary increase in energy and heightened senses, followed by an unprecedented boost in physical and '
             'mental capacity. However, these symptoms have not been observed or documented in any medical '
             'facility.Furthermore, images circulating online of individuals purportedly affected by ZX-5 show '
             'exaggerated physical transformations, such as glowing eyes and unusual skin patterns, which experts '
             'have dismissed as digital manipulations.Authorities are urging the public to rely on trusted sources '
             'for COVID-19 information and to disregard these latest claims about ZX-5. Vaccination and adherence to '
             'public health guidelines remain the best defense against the virus.As of now, the countries with the '
             'highest reported cases of this fictional ZX-5 variant include Atlantis (a mythical island), '
             'Utopia (a fictional society), and El Dorado (a legendary city of gold), highlighting the fantastical '
             'nature of these reports In conclusion, the ZX-5 variant remains a fabricated concept with no basis in '
             'reality, serving as a stark reminder of the power of misinformation in the digital age.')
result = preprocess_and_predict(fake_news)
print(result)
