import pandas as pd
import numpy as np
import re
import seaborn as sns
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def translate_personality(personality):
    mbti_binary = {'I':0, 'E':1, 'N':0, 'S':1, 'T':0, 'F':1, 'J':0, 'P':1}
    # transform mbti to binary vector
    return [mbti_binary[l] for l in personality]

def translate_back(personality):
    mbti = [{0:'I', 1:'E'}, {0:'N', 1:'S'}, {1:'F', 0:'T'}, {0:'J', 1:'P'}]
    # transform binary vector to mbti personality
    s = ""
    for i, l in enumerate(personality):
        s += mbti[i][l]
    return s

def preprocess_data(data):
    # Lemmatize
    stemmer = PorterStemmer()
    lemmatiser = WordNetLemmatizer()
    stopWords = stopwords.words("english")
    mbti_words =  ['infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp', 'isfp', 'istp', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj']
    list_personality = []
    list_posts = []

    for row in data.iterrows():
        posts = row[1].posts
        #remove url links
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)
        #remove special characters and numbers
        temp = re.sub("[^a-zA-Z]", " ", temp)
        #remove extra spaces
        temp = re.sub(' +', ' ', temp).lower()
        #remove stopwords
        temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in stopWords])
        
        #remove mbti personality words
        for t in mbti_words:
            temp = temp.replace(t,"")

        type_labelized = translate_personality(row[1].type)
        list_personality.append(type_labelized)
        list_posts.append(temp)

    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality
                    
def tfIdfRepresentation(list_posts):
    # Posts to a matrix of token counts
    cntizer = CountVectorizer(analyzer="word", 
                                max_features=1500, 
                                tokenizer=None,    
                                preprocessor=None, 
                                stop_words=None,  
                                max_df=0.7,
                                min_df=0.1) 

    # Learn the vocabulary dictionary and return term-document matrix
    print("CountVectorizer...")
    X_cnt = cntizer.fit_transform(list_posts)

    # Transform the count matrix to a normalized tf or tf-idf representation
    tfizer = TfidfTransformer()

    print("Tf-idf...")
    # Learn the idf vector (fit) and transform a count matrix to a tf-idf representation
    X_tfidf =  tfizer.fit_transform(X_cnt).toarray()
    return X_tfidf


def loadModel(loadPretrainedModel,type_indicator):
    model=None
    if(loadPretrainedModel==False):
        #load dataset
        data = pd.read_csv('./data/mbti_1.csv') 
        list_posts, list_personality  = preprocess_data(data)
        print("Num posts and personalities: ",  list_posts.shape, list_personality.shape)

        # setup parameters for xgboost
        param = {}
        param['n_estimators'] = 1000
        param['max_depth'] = 10
        param['nthread'] = 8
        param['learning_rate'] = 0.01
        param['subsample'] = 0.8
        param['colsample_bytree'] = 1
        param['gamma'] = 1
        param['reg_alpha']=0.3
        param['scale_pos_weight']=1

        type_indicators = [ "IE", "NS", "TF", "JP"  ]

        X = tfIdfRepresentation(list_posts)
       
        print("%s ..." % (type_indicator))
        
        Y = list_personality[:,type_indicators.get(type_indicator)]
        # split data into train and test sets
        seed = 7
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
        # fit model on training data
        model = XGBClassifier(**param,objective='binary:logistic')
        model.fit(X_train, y_train)
        
        # Save xgb_params for later discussuin
        default_get_xgb_params = model.get_xgb_params()
        print (default_get_xgb_params)
    else:
        file_name = "personalit{}.pkl".format()
        model = pickle.load(open(file_name, "rb"))

    return model

