#import spotipy
#from spotipy.oauth2 import SpotifyClientCredentials
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from nltk.tokenize import word_tokenize
import numpy as np
import pickle

nltk.download('stopwords')
nltk.download('punkt')

classifier_file = 'nb_clf'
cv_file = 'cv_vec'
tfidf_file = 'tfidf_vec'

label_mapping = ['R&B', 'Hip-Hop', 'Rock', 'Country', 'Jazz', 'K-Pop', 'Reggae', 'Gospel', 'Funk']

no_html_tags = re.compile('<.*?>') 

def lyrics_remove_html(text):
  cleantext = re.sub(no_html_tags, '', text)
  return cleantext

def get_train_data(path):
    genres = ['R&B', 'Hip-Hop', 'Rock', 'Country', 'Jazz', 'K-Pop', 'Reggae', 'Gospel', 'Funk']
    with open(path, 'r') as fl:
        all_lines = fl.readlines()
        lyrics = ''
        training_data = []
        count = 0
        for line in all_lines:
            text_wo_newl = line.strip('\n')

            if text_wo_newl in genres:
                training_data.append([lyrics, text_wo_newl])
                lyrics = ''
                print('Genre: {}, Count: {}'.format(text_wo_newl, count))
                count = 0
            elif text_wo_newl != '':
                count += 1
                lyrics += text_wo_newl + ' '

        return training_data


#train Naive Bayes Classifier to predict genre based on lyrics 
def train_naive_bayes(input_data):
    #Perform stop word removal, punctuation removal, stemming, on each row
    count_vec = CountVectorizer()
    tf_idf_trans = TfidfTransformer()
    stemmer = PorterStemmer()

    np_input_data = np.array(input_data)
    for train_row in np_input_data:
        text = train_row[0]
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english') and not word in string.punctuation]
        stemmed_tokens = [stemmer.stem(word) for word in tokens_without_sw]
        cleaned_lyrics = ' '.join(stemmed_tokens)
        train_row[0] = cleaned_lyrics

    clf = GaussianNB()

    train_lyrics = np_input_data[:,0]
    # labels below map to -> ['R&B', 'Hip-Hop', 'Rock', 'Country', 'Jazz', 'K-Pop', 'Reggae', 'Gospel', 'Funk']
    train_genre = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    genre_term_count = count_vec.fit_transform(train_lyrics)
    term_count_tfidf = tf_idf_trans.fit_transform(genre_term_count)

    clf.fit(term_count_tfidf.toarray(), train_genre)

    #save Naive Bayes model
    pickle.dump(clf, open(classifier_file, 'wb'))
    pickle.dump(count_vec, open(cv_file, 'wb'))
    pickle.dump(tf_idf_trans, open(tfidf_file, 'wb'))

def predict_genre(lyrics):
    count_vec = pickle.load(open(cv_file, 'rb'))
    tf_idf_trans = pickle.load(open(tfidf_file, 'rb'))
    stemmer = PorterStemmer()

    for text in lyrics:
        text_tokens = word_tokenize(text)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english') and not word in string.punctuation]
        stemmed_tokens = [stemmer.stem(word) for word in tokens_without_sw]
        cleaned_lyrics = ' '.join(stemmed_tokens)
        text = cleaned_lyrics

    test_lyrics_cv = count_vec.transform(lyrics)
    tf_idf_test_lyrics = tf_idf_trans.transform(test_lyrics_cv)

    #load Naive Bayes model
    nb_classifier = pickle.load(open(classifier_file, 'rb'))
    predicted = nb_classifier.predict(tf_idf_test_lyrics.toarray())
    return label_mapping[predicted[0]]



    

if __name__ == "__main__":
    get_train_data('lyrics_train_data.txt')        
        


