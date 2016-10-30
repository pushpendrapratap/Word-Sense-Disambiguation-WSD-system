import A
from sklearn.feature_extraction import DictVectorizer

################################################################   # Courtesy Pushpendra pratap
import nltk                      
import collections
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer  # This stemmer is better for English lang. than Porter Stemmer.
#from nltk.stem.porter import *
from sklearn import svm
from sklearn import neighbors
from nltk.collocations import *
# from nltk.corpus import wordnet as wn
# import re
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from nltk.probability import *
################################################################

# You might change the window size
window_size = 10 # 15

############################################################################### Courtesy Pushpendra pratap
def stemming_and_stop_words(language):
    lang = language.lower()  
    
    if (lang=='spanish' or lang=='catalan'):     
        lang = 'spanish'
        tokenizer = nltk.data.load('nltk:tokenizers/punkt/spanish.pickle')   

    elif(lang=='english'):
        tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle') 

    punctuations = list(string.punctuation)
    temp_stop = stopwords.words(lang)

    stop = set(punctuations + temp_stop)
    stemmer = SnowballStemmer(lang)

    return stemmer, stop, tokenizer
###############################################################################


# B.1.a,b,c,d
def extract_features(data, stemmer, stop, tokenizer):   ##################
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

    ########################################################################### # Courtesy Pushpendra pratap
    # obj = re.search(r'([a-zA-Z0-9]*)+(\.[a-zA-Z0-9]*)', data[0][0])
    # obj = re.search(r'([^.]*)+(\.[^.]*)', data[0][0])
    # match_str = obj.group()
    
    # temp = []                                           
    # for ss in wn.synsets(match_str):                          # i.e., head keyword in wn.synsets()
    #     temp = temp + nltk.word_tokenize(ss.definition())
    # temp = [j for j in temp if j not in stop]
    # temp = [stemmer.stem(plural) for plural in temp]
    # temp = set(temp)

    bigram_measures = nltk.collocations.BigramAssocMeasures()

    # implement your code here      ###########################################
    for i in data:
        wi = []

        left = tokenizer.tokenize(i[1])
        right = tokenizer.tokenize(i[3])

        if(len(left)==0 or len(right)==0):
            continue

        left_context_tokenize = nltk.word_tokenize(left[-1])
        right_context_tokenize = nltk.word_tokenize(right[0])

        # left_context_tokenize = nltk.word_tokenize(i[1])
        # right_context_tokenize = nltk.word_tokenize(i[3])

        head = nltk.word_tokenize(i[2])

        wi = left_context_tokenize[-(min(window_size, len(left_context_tokenize))) : ] + \
             head + right_context_tokenize[ : (min(window_size, len(right_context_tokenize)))]

        # for k in range(len(wi)):    
        #     if wi[k].isdigit():
        #         wi[k] = '_NUMBER_' 

        finder = BigramCollocationFinder.from_words(wi)
        # wi = wi + finder.nbest(bigram_measures.pmi, 2)

        # colloc1 = left_context_tokenize[-(min(2, len(left_context_tokenize))) : ] + \
        #            head + right_context_tokenize[ : (min(2, len(right_context_tokenize)))]

        # wi = [j for j in wi if j not in stop]             # removed stop words and punctuations .

        # wi = wi + colloc1

        wi = [stemmer.stem(plural) for plural in wi]      # stemming of words .
        wi = wi + finder.nbest(bigram_measures.pmi, 3)
        wi = list(set(wi))

        # wi_with_pos_tag = nltk.pos_tag(wi)                # POS tagging 
        # pos_token_list = []
        # for j in wi_with_pos_tag:
        #     pos_token_list.append(j[0])
        #     pos_token_list.append(j[1])
        # pos_token_list = pos_token_list + finder.nbest(bigram_measures.pmi, 2)

        # colloc_pos_list = finder.nbest(bigram_measures.pmi, 2)
        # for j in colloc_pos_list:
        #     wi_with_pos_tag = wi_with_pos_tag + nltk.pos_tag(j)  


        features[i[0]] = collections.Counter(wi)  

        # features[i[0]] = collections.Counter(wi_with_pos_tag)
        # features[i[0]] = collections.Counter(pos_token_list)

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
    X_train = {i[0]:i[1] for i in zip(train_features.keys(), vec.transform(train_features.values()).toarray())}
    X_test = {i[0]:i[1] for i in zip(test_features.keys(), vec.transform(test_features.values()).toarray())}

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



    # implement your code here   ################################################################
    # new_y_train = []    
    # for i in X_train.keys():
    #     new_y_train.append(y_train[i])

    # model = SelectKBest(chi2, k='all').fit(X_train.values(), new_y_train) 

    # model = SelectKBest(chi2, k=5).fit(X_train.values(), new_y_train)
    # X_train_new = dict(zip( X_train.keys(), model.transform(X_train.values()) )) 
    # X_test_new = dict(zip( X_test.keys(), model.transform(X_test.values()) ))   
    #############################################################################################

    # return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train, X_test

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

    new_y_train = []               #  since, python Dictionary used to be unordered .
    for i in X_train.keys():
        new_y_train.append(y_train[i])

    svm_clf.fit(X_train.values(), new_y_train)
    # knn_clf.fit(X_train.values(), new_y_train)

    svm_results = zip(X_test.keys(), svm_clf.predict(X_test.values()))
    # knn_results = zip(X_test.keys(), knn_clf.predict(X_test.values()))

    results = svm_results[:]     # check for knn_results also that which give better results.
    # results = knn_results[:]
    ###########################################################################################

    return results

# run part B
def run(train, test, language, answer):
    results = {}
    stemmer, stop, tokenizer = stemming_and_stop_words(language)         # Courtesy Pushpendra pratap

    for lexelt in train:
        train_features, y_train = extract_features(train[lexelt], stemmer, stop, tokenizer)   # Courtesy Pushpendra pratap
        test_features, _ = extract_features(test[lexelt], stemmer, stop, tokenizer)     # Courtesy Pushpendra pratap

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    answer = answer + '-' + language    # Courtesy Pushpendra pratap

    A.print_results(results, answer)