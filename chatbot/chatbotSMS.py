#import what is required for sms messaging
from flask import Flask, request
import requests
from twilio.twiml.messaging_response import MessagingResponse

#Import the relevant nlp libraries needed to generate responses.
import nltk
import numpy
import random
import string
# Additional training
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

chatbots_file = open('chatbot.txt','r',errors = 'ignore')
content = chatbots_file.read()
content = content.lower()
#nltk.download('punkt')
#nltk.download('wordnet')
sentence_tokens = nltk.sent_tokenize(content)
word_tokens = nltk.word_tokenize(content)

#Pre-Processing
lemmer = nltk.stem.WordNetLemmatizer()

def lem_tokens(tokens):
    return[lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def lem_normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
    
app = Flask(__name__)

# User Input
GREETING_INPUTS = ("hello","hi","what's up","sup","hey")
GREETING_RESPONSES = ("Hi! What would you like to know?","Hey! What would you like to know?","Hi there! What would you like to know?","Hello! What would you like to know?")

def greeting(sentence):
    for word in sentence.split(" "):
        if request.values.get('Body', '').lower() in GREETING_INPUTS:
            greeting = random.choice(GREETING_RESPONSES)
            return greeting

#Define sms incoming message

def response(user_response):

    robo_response = ''
    sentence_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer = lem_normalize, stop_words = 'english')
    tfidf = TfidfVec.fit_transform(sentence_tokens)
    
    values = cosine_similarity(tfidf[-1],tfidf)
    idx = values.argsort()[0][-2]
    flat = values.flatten()
    flat.sort()

    req_tfidf = flat[-2]
    
    if(req_tfidf==0):
        robo_response = robo_response + "I am sorry! I don't understand you. "
    else:
        robo_response = robo_response + sentence_tokens[idx]

flag = True
print("Indica: My name is Indica. Ask me about the market, and all your agriculture needs. If you want to exit, type bye!")
#Allow user to exit out from bot responses, and add addtional question clarifications.

@app.route('/bot', methods=['POST'])

def parsing():
    while(flag==True):
        user_response = input().lower()
        #user_response = request.values.get('Body', '').lower()
        resp = MessagingResponse()
        msg = resp.message()
        responded = False

        #Add extra questions for question subcategories
        if(user_response == 'insecticides'):
            print("Indica: What about insecticides do you want information about? [Price, Type]")
            msg.body("Indica: What about insecticides do you want information about? [Price, Type]")
            user_response = 'insecticides' + user_response
        
        if(user_response!='bye'):
            if(user_response=='thanks' or user_response=='thank you'):
                flag = False
                print("Indica: You are welcome!")
                msg.body("Indica: You are welcome!")
            else:
                if(greeting(user_response)!=None):
                    print("Indica: "+greeting(user_response))
                    msg.body("Indica: "+greeting(user_response))
                else:
                    print("Indica: " +response(user_response))
                    msg.body("Indica: " +response(user_response))
                    sentence_tokens.remove(user_response)
        else:
            flag = False
            msg.body("Indica: Bye, take care! Let me know if you need help.")

if __name__ == "__main__":
  app.run()


