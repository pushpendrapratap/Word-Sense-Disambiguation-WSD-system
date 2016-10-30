from main import replace_accented
from sklearn import svm
from sklearn import neighbors

##################################################################### # Courtesy Pushpendra pratap
import collections                   # Courtesy Pushpendra pratap
import nltk
import codecs
#####################################################################

# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = {}

    # implement your code here   #############################################
    for lexelt in data:
        wi = []
        for i in data[lexelt]:
            left_context_tokenize = nltk.word_tokenize(i[1])
            right_context_tokenize = nltk.word_tokenize(i[3])
            wi = wi + left_context_tokenize[-(min(window_size, len(left_context_tokenize))) : ] + \
                 right_context_tokenize[ : (min(window_size, len(right_context_tokenize)))]         
            wi = list(set(wi))
        # wi.sort()  
        s[lexelt] = wi[:]
    ###########################################################################

    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}

    # implement your code here     ########################################          
    len_s = len(s)

    for i in data:
        temp = [0]*len_s          # initialize a list of size len_s and all its elements will be initially 0 .
        wi = []

        left_context_tokenize = nltk.word_tokenize(i[1])
        right_context_tokenize = nltk.word_tokenize(i[3])

        wi = left_context_tokenize[-(min(window_size, len(left_context_tokenize))) : ] + \
               right_context_tokenize[ : (min(window_size, len(right_context_tokenize)))]

        wi = list(set(wi))

        for j in wi:
            if j in s:
                temp[s.index(j)] = 1

        vectors[i[0]] = temp[:]

        labels[i[0]] = i[4]
    ##########################################################################

    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

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

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()

    # implement your code here     ############################################
    svm_clf.fit(X_train.values(), y_train.values())
    knn_clf.fit(X_train.values(), y_train.values())

    svm_results = zip(X_test.keys(), svm_clf.predict(X_test.values()))
    knn_results = zip(X_test.keys(), knn_clf.predict(X_test.values()))
    ############################################################################

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here 
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing  ##################################
    outfile = codecs.open(output_file + '.answer', encoding='utf-8', mode='w')
    for lexelt, instances in sorted(results.iteritems(), key=lambda d: replace_accented(d[0].split('.')[0])):
        for instance in sorted(instances, key=lambda d: int(d[0].split('.')[-1])):
            instance_id = instance[0]
            sid = instance[1]
            outfile.write(replace_accented(lexelt + ' ' + instance_id + ' ' + sid + '\n'))
    outfile.close()
    #############################################################################################

# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    svm_file = svm_file + '-' + language     # courtesy Pushpendra pratap
    knn_file = knn_file + '-' + language     # courtesy Pushpendra pratap

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



