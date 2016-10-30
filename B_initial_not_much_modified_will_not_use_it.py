import A
from sklearn.feature_extraction import DictVectorizer

######################################################################  # Courtesy Pushpendra pratap
import nltk                      # Courtesy Pushpendra pratap
import collections
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer  # This stemmer is better for English lang. than Porter Stemmer.
from sklearn import svm
from sklearn import neighbors
from nltk.collocations import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from nltk.probability import *
######################################################################


# You might change the window size
window_size = 15

# B.1.a,b,c,d
def extract_features(data):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}

    # implement your code here  
    for i in data:
        wi = []
        left_context_tokenize = nltk.word_tokenize(i[1])
        right_context_tokenize = nltk.word_tokenize(i[3])

        wi = left_context_tokenize[-(min(window_size, len(left_context_tokenize))) : ] + \
              right_context_tokenize[ : (min(window_size, len(right_context_tokenize)))]

        wi = list(set(wi)) 
        features[i[0]] = collections.Counter(wi)
        labels[i[0]] = i[4]

    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''


    # implement your code here   
    model = SelectKBest(chi2, k=5).fit(X_train.values(), y_train.values())
    X_train_new = dict(zip( X_train.keys(), model.transform(X_train.values()) )) 
    X_test_new = dict(zip( X_test.keys(), model.transform(X_test.values()) ))   

    return X_train_new, X_test_new
    # or return all feature (no feature selection):
    # return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    results = []


    # implement your code here    #############################################################
    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()

    svm_clf.fit(X_train.values(), y_train.values())
    # knn_clf.fit(X_train.values(), y_train.values())

    svm_results = zip(X_test.keys(), svm_clf.predict(X_test.values()))
    # knn_results = zip(X_test.keys(), knn_clf.predict(X_test.values()))

    results = svm_results[:]     # check for knn_results also that which give better results.
    # results = knn_results[:]
    ###########################################################################################

    return results

# run part B
def run(train, test, language, answer):
    results = {}

    for lexelt in train:
        train_features, y_train = extract_features(train[lexelt])
        test_features, _ = extract_features(test[lexelt])

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    answer = answer + '-' + language    # Courtesy Pushpendra pratap
    A.print_results(results, answer)