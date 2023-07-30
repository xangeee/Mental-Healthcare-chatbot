import tensorflow
import numpy as np
import pandas as pd
import json
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM , Dense,Flatten
from tensorflow.keras.models import Model
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer, WordNetLemmatizer
import personalityModel as personality
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import snscrape.modules.twitter as sntwitter
import tweepy
from xgboost import XGBClassifier
import sys
import subprocess
import threading 

def preprocessData(df, col,stop):
    df['preprocessed'+col] = df[col].apply(lambda x : " ".join([word for word in x.split(" ") if word not in stop]))
    df['preprocessed'+col] = df['preprocessed'+col].str.replace('[^a-zA-Z0-9 ]', '')
    df['preprocessed'+col] = df['preprocessed'+col].str.lower()
    return df

def getModel(trainModel):
    with open('./data/intents.json') as intents:
        data1 = json.load(intents)

    tags = []
    inputs = []
    responses={}

    for intent in data1['intents']:
        responses[intent['tag']]=intent['responses']
        for lines in intent['input']:
            inputs.append(lines)
            tags.append(intent['tag'])
            

    data = pd.DataFrame({"inputs":inputs,"tags":tags})
    data = data.sample(frac=1)

    #Mental Health Question Answers
    mh = pd.read_csv('./data/Mental_Health_FAQ.csv')

    dataAux = list(zip(mh.Question_ID, mh.Answers))
    for i in dataAux:
        responses[str(i[0])]=[i[1]]

    dataMh=pd.DataFrame({"inputs":mh.Questions.tolist(),"tags":[str(i) for i in mh.Question_ID.tolist()]})
    dataMh = dataMh.sample(frac=1)

    data=data.append(dataMh, ignore_index=True)

    stop = stopwords.words('english')
    data = preprocessData(data, 'inputs',stop)

    data['preprocessedinputs'] = data['preprocessedinputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
    data['preprocessedinputs'] = data['preprocessedinputs'].apply(lambda wrd: ''.join(wrd))
    data['preprocessedinputs']

    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(data['preprocessedinputs'])
    train = tokenizer.texts_to_sequences(data['preprocessedinputs'])
    x_train = pad_sequences(train)
    le = LabelEncoder()
    y_train = le.fit_transform(data['tags'])
    input_shape = x_train.shape[1]
     #define vocabulary
    vocabulary = len(tokenizer.word_index)
    #print("number of unique words : ",vocabulary)
    output_length = le.classes_.shape[0]
    #print("output length: ",output_length)
    
    if trainModel:
        model=createModel(input_shape,output_length,vocabulary,x_train,y_train)
        train = model.fit(x_train,y_train,epochs=250)
        model.save('chatModel.hdf5')
    else:
        model=keras.models.load_model('chatModel.hdf5')

    return model,tokenizer,input_shape,responses,le

def  createModel(input_shape,output_length,vocabulary,x_train,y_train):
    #creating the model
    i = Input(shape=(input_shape,))
    x = Embedding(vocabulary+1,10)(i)
    x = LSTM(15,return_sequences=True)(x)
    
    x = Flatten()(x)
    x = Dense(output_length,activation="softmax")(x)
    model = Model(i,x)
    
    model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
    return model

def get_tweets(username):
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'from:{username}').get_items()):
        tweets.append(tweet.content)
        if i == 1000: break
    return tweets

def preprocess_tweets(tweets):
    # Lemmatize
    stemmer = PorterStemmer()
    lemmatiser = WordNetLemmatizer()
    stopWords = stopwords.words("english")
    mbti_words =  ['infj', 'entp', 'intp', 'intj', 'entj', 'enfj', 'infp', 'enfp', 'isfp', 'istp', 'isfj', 'istj', 'estp', 'esfp', 'estj', 'esfj']

    #remove url links
    temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', tweets)
    #remove special characters and numbers
    temp = re.sub("[^a-zA-Z]", " ", tweets)
    #remove extra spaces
    temp = re.sub(' +', ' ', temp).lower()
    #remove stopwords
    temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in stopWords])

    #remove mbti personality words
    for t in mbti_words:
        temp = temp.replace(t,"")

    posts = np.array([temp])
    return posts

def tfIdfRepresentation(list_posts):
    # Posts to a matrix of token counts
    cntizer = CountVectorizer(analyzer="word", 
                                max_features=200000, 
                                tokenizer=None,    
                                preprocessor=None, 
                                stop_words=None,  
                                max_df=1,
                                min_df=1) 

    # Learn the vocabulary dictionary and return term-document matrix
    print("CountVectorizer...")
    X_cnt = cntizer.fit_transform(list_posts)

    # Transform the count matrix to a normalized tf or tf-idf representation
    tfizer = TfidfTransformer()

    print("Tf-idf...")
    # Learn the idf vector (fit) and transform a count matrix to a tf-idf representation
    X_tfidf =  tfizer.fit_transform(X_cnt).toarray()
    return X_tfidf



def personalityPrediction(username,loadPretrainedModel):
    # your bearer token
    MY_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAIzMcwEAAAAAALWWof82iRlW9j%2B4vOa83Ee7ZKM%"
    # create your client 
    client = tweepy.Client(bearer_token=MY_BEARER_TOKEN)
    tweets=get_tweets(username)
    list_posts=preprocess_tweets(tweets)
    X_tfidf=tfIdfRepresentation(list_posts)
    
    mbti=""
    model=personality.loadModel(loadPretrainedModel,"IE")
    y_pred=model.predict(X_tfidf)
    mbti+='E' if [round(value) for value in y_pred]==1 else 'I'
   
    model=personality.loadModel(loadPretrainedModel,"NS")
    model.predict(X_tfidf)
    y_pred=model.predict(X_tfidf)
    mbti+='S' if [round(value) for value in y_pred]==1 else 'N'
        
    model=personality.loadModel(loadPretrainedModel,"TF")
    model.predict(X_tfidf)
    y_pred=model.predict(X_tfidf)
    mbti+='F' if [round(value) for value in y_pred]==1 else 'T'
        
    model=personality.loadModel(loadPretrainedModel,"JP")
    model.predict(X_tfidf)
    y_pred=model.predict(X_tfidf)
    mbti+='P' if [round(value) for value in y_pred]==1 else 'J'
    
    return mbti

import tkinter as tk

def chat(trainModel=False):
    model,tokenizer,input_shape,responses,le=getModel(trainModel)
    for i in range(1,10):
        #texts_p = []
        name_var=tk.StringVar()
        #prediction_input = input('Me : ')
        l1=tk.Label(root, text="Say something : ", foreground='black', bg='white').place(x=10, y=280)
        sName=tk.Entry(root)
        sName.place(x=110, y=280, width=250)
        submitBtn=tk.Button(root, padx=0, pady=0, text="Send", bd=0, foreground='white', bg='black')
        submitBtn.config(command=lambda: predict(sName.get()))
        submitBtn.place(x=390, y=280, width=140)
        sName.delete()
        
        
def predict(prediction_input):
    print("Me : ",prediction_input)
    if prediction_input == "personality" or prediction_input=='mbti':
        twitter_user = input('Provide a twitter user : ')
        print("Dolores : Your personality is: ",personalityPrediction(twitter_user,True))
       
    else:
        #removing punctuation and converting to lowercase
        prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
        prediction_input = ''.join(prediction_input)
        texts_p.append(prediction_input)
        #tokenizing and padding
        prediction_input = tokenizer.texts_to_sequences(texts_p)
        prediction_input = np.array(prediction_input).reshape(-1)
        prediction_input = pad_sequences([prediction_input],input_shape)
        #getting output from model
        output = model.predict(prediction_input)
        output = output.argmax()
        #finding the right tag and predicting
        response_tag = le.inverse_transform([output])[0]
        print("Dolores : ",random.choice(responses[response_tag]))
        print("\n")
        #if response_tag == "goodbye":
         #   break
        
def run():
    threading.Thread(target=chat).start()
    
class Redirect():

    def __init__(self, widget, autoscroll=True):
        self.widget = widget
        self.autoscroll = autoscroll

    def write(self, text):
        self.widget.insert('end', text)
        if self.autoscroll:
            self.widget.see("end")  # autoscroll
        
    def flush(self):
        pass

texts_p = []
root = tk.Tk()
model,tokenizer,input_shape,responses,le=getModel(False)
# - Frame with Text and Scrollbar -

frame = tk.Frame(root)
frame.pack(expand=True, fill='both')

text = tk.Text(frame)
text.pack(side='left', fill='both', expand=True)

scrollbar = tk.Scrollbar(frame)
scrollbar.pack(side='right', fill='y')

text['yscrollcommand'] = scrollbar.set
scrollbar['command'] = text.yview

old_stdout = sys.stdout    
sys.stdout = Redirect(text)

# - rest -

button = tk.Button(root, text='Talk to me', command=run)
button.pack()

root.mainloop()

# - after close window -

sys.stdout = old_stdout

            

            
                

