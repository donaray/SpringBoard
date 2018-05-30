#import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from collections import defaultdict, Counter
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree, linear_model, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import r2_score, accuracy_score, fbeta_score, roc_auc_score
from sklearn.model_selection import cross_val_score
#from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification
from sklearn.metrics import classification_report


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler

from sklearn import grid_search
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.cross_validation import train_test_split, ShuffleSplit

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from textblob import TextBlob
from textblob import Word

pd.options.display.max_colwidth = 500
#from nltk.autocorrect import spell


#Read the dataset, 
df_all = pd.read_csv('../data/Clothing.csv')

# Drop rows with missing data
df_nonmissing = df_all[~df_all['Review Text'].isnull()]

#df_sub = df_nonmissing[['Rating', 'Department Name', 'Division Name', 'Class Name']]

df_nonmissing.columns = ['Index', 'ClothingID', 'Age', 'Title', 'ReviewText', 'Rating', 'RecommendedIND', 
                         'PositiveFeedbackCount','DivisionName', 'DepartmentName', 'ClassName']


def Print_Stats(df, num):
    #Prints basic summary statistics of a pandas dataframe    

    print(df.head(num))
    print(df.shape)
    print(df.info())
    print(df.columns)
    print(df.describe())
    
def Print_Frequency(df, bins = []):
    #Prints frequency distribution
    
    col_list = list(df.columns.values)
    
    if bins == []:
        for col in col_list:
            print(df[col].value_counts().sort_index())
    else:
        for col in col_list:
            print(pd.cut(df[col],bins).value_counts().sort_index())


# Cleaning Text Data

df = df_nonmissing

df['word_count'] = df['ReviewText'].apply(lambda x: len(str(x).split(" ")) )

df['char_count'] = df['ReviewText'].str.len()

print(df[['ReviewText', 'word_count', 'char_count']].head())

print(df['word_count'].sum())
print(df['word_count'].mean())


# Stopwords
import nltk
#nltk.download()
from nltk.corpus import stopwords
stop = stopwords.words('english')

#stop = ['and']

df['stopwords'] = df['ReviewText'].apply(lambda x: len([x for x in x.split() if x in stop]))

df['numerics'] = df['ReviewText'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

df['upper'] = df['ReviewText'].apply(lambda x: len([x for x in x.split() if x.isupper()]))


#df['Correct'] = df['ReviewText'].apply(lambda x: str(TextBlob(x).correct()))

#df['word_split'] = df['ReviewText'].apply(lambda x: str(x).split(" ") )

word_freq = pd.Series(' '.join(df['ReviewText']).split()).value_counts()

#rare_words = pd.Series(' '.join(df['ReviewText']).split()).value_counts()[-500:]

values_list = dict(word_freq)
values_list = Counter(values_list)

for k, v in values_list.most_common(10):
    print(k, v)
    
# Print the 10 most common words
count_once = 0
for k, v in values_list.items():
    if v == 1:
        #print(k, v)
        count_once += 1

print('Total number of words that appear once: ', count_once)      

print('Total word count')
print(df['ReviewText'].apply(lambda x: len(str(x).split(" ")) ).sum())

print('Total number of unique words')

uniqueWords = list(set(" ".join(df['ReviewText']).lower().split(" ")))
count = len(uniqueWords)

print(count)

# Convert to lower case
df['ReviewText'] = df['ReviewText'].apply(lambda x: " ".join(x.lower() for x in x.split()))

print('Convert to lower case\n----')
print('Total word count')
print(df['ReviewText'].apply(lambda x: len(str(x).split(" ")) ).sum())
print(" ")
print('Total number of unique words')
uniqueWords = list(set(" ".join(df['ReviewText']).lower().split(" ")))
count = len(uniqueWords)
print(count)
print('')


# Remove Punctuation
df['ReviewText'] = df['ReviewText'].str.replace('[^\w\s]','')

print('Remove Punctuation\n----')
print('Total word count')
print(df['ReviewText'].apply(lambda x: len(str(x).split(" ")) ).sum())
print(" ")
print('Total number of unique words')
uniqueWords = list(set(" ".join(df['ReviewText']).lower().split(" ")))
count = len(uniqueWords)
print(count)
print('')


# Remove Stopwords
df['ReviewText'] = df['ReviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


print('Remove Stopwords\n----')
print('Total word count')
print(df['ReviewText'].apply(lambda x: len(str(x).split(" ")) ).sum())
print(" ")
print('Total number of unique words')
uniqueWords = list(set(" ".join(df['ReviewText']).lower().split(" ")))
count = len(uniqueWords)
print(count)
print('')

# Remove numerics
df['ReviewText'] = df['ReviewText'].apply(lambda x: " ".join(x for x in x.split() if not x.isdigit()))

print('Remove Numerics\n----')
print('Total word count')
print(df['ReviewText'].apply(lambda x: len(str(x).split(" ")) ).sum())
print(" ")
print('Total number of unique words')
uniqueWords = list(set(" ".join(df['ReviewText']).lower().split(" ")))
count = len(uniqueWords)
print(count)
print('')

# Change to root word
df['ReviewText'] = df['ReviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


print('Stem Words\n----')
print('Total word count')
print(df['ReviewText'].apply(lambda x: len(str(x).split(" ")) ).sum())
print(" ")
print('Total number of unique words')
uniqueWords = list(set(" ".join(df['ReviewText']).lower().split(" ")))
count = len(uniqueWords)
print(count)
print('')


# Most Common and Rare words
word_freq = pd.Series(' '.join(df['ReviewText']).split()).value_counts()

values_list = dict(word_freq)
values_list = Counter(values_list)
    
# Print the 10 most common words
count_once = 0
rare_words = []
for k, v in values_list.items():
    if v == 1:
        #print(k, v)
        rare_words.append(k)
        count_once += 1

print('Total number of words that appear once: ', count_once)      
print("10 most rare words")
print(rare_words[0:10])

# Remove words that appear once: rare words, missplet words
df['ReviewText'] = df['ReviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in rare_words))


print('Remove words that appear once\n----')
print('Total word count')
print(df['ReviewText'].apply(lambda x: len(str(x).split(" ")) ).sum())
print(" ")
print('Total number of unique words')
uniqueWords = list(set(" ".join(df['ReviewText']).lower().split(" ")))
count = len(uniqueWords)
print(count)
print('')

print("10 Most Common Words")
for k, v in values_list.most_common(10):
    print(k, v)
    
# MODELLING

# Define a function to fit and print results

def Model(X_train, y_train, X_test, y_test, classifier):
    reg = classifier.fit(X_train, y_train)
    
    # Compute metrics
    y_pred = reg.predict(X_test)
    score = r2_score(y_test, y_pred)
    score_1 = reg.score(X_test, y_test)
    accuracy_test = accuracy_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, reg.predict(X_train))
    
    print('Accuracy score on training data: {:.4f}'.format(accuracy_train))
    print('Accuracy score on testing data: {:.4f}'.format(accuracy_test))
    print('')    
    
    
    print('Roc score: {:.4f}'.format(roc_score))
    print('')

    # Generate the confusion matrix and classification report
    print('Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    print('')
    
    print('Classification Report')
    print(classification_report(y_test, y_pred))

    return reg


# Define a pipeline combining a text feature extractor with a Random Forest classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clfRF', RandomForestClassifier(class_weight='balanced'))
])


# Define Parameter Space
parameters = {
    'vect__max_df': (0.5, 0.75),
    'vect__min_df' : (0, 1, 5, 10),
    #'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'clfRF__n_estimators': (5, 10, 25, 50, 100),
}


data = [4, 5]
y = [int(n in set(data)) for n in df['Rating']]

# Make an fbeta_score scoring object
#scorer = make_scorer(accuracy_score) 

# Split the data into train test data
X_train, X_test, y_train, y_test = train_test_split(df['ReviewText'], y, test_size=0.2, random_state = 99)

# find the best parameters for both the feature extraction and the classifier
clf_RF_countVect = GridSearchCV(pipeline, parameters)


print('Results using CountVectorizer and Random Forest\n------')
reg_RF_countVect = Model(X_train, y_train, X_test, y_test, clf_RF_countVect)

print('Random Forest  with Count Vectorizer\n-----')
print(clf_RF_countVect.best_params_)

