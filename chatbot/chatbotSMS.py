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

import os

NAME = "Chikoo"
chatbots_file = open('chatbot.txt','r')#, errors = 'ignore')
content = chatbots_file.read()
content = content.lower()
#nltk.download('punkt')
#nltk.download('wordnet')
sentence_tokens = nltk.sent_tokenize(content)
word_tokens = nltk.word_tokenize(content)

#Pre-Processing
lemmer = nltk.stem.WordNetLemmatizer()

# User Input
GREETING_INPUTS = ("hello","hi","what's up","sup","hey")
GREETING_RESPONSES = ("Hi! What would you like to know?","Hey! What would you like to know?","Hi there! What would you like to know?","Hello! What would you like to know?")

def lem_tokens(tokens):
    return[lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def lem_normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


class SmsResponse:
    state = 0 # 0=start, 1=insecticides, -1=done
    @staticmethod
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
        
        print("")
        return robo_response

    @staticmethod
    def greeting(request, sentence):
        for word in sentence.split(" "):
            if request.values.get('Body', '').lower() in GREETING_INPUTS:
                greet = random.choice(GREETING_RESPONSES)
                return greet



#Allow user to exit out from bot responses, and add addtional question clarifications.
print("Chikoo: My name is Indica. Ask me about the market, and all your agriculture needs. If you want to exit, type bye")
app = Flask(__name__)

@app.route('/bot', methods=['POST'])
def bot():
    user_response = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()
    responded = False

    if SmsResponse.state == -1:
        msg.body("Chikoo deactivated already.")
    else:
        if SmsResponse.state == 0:
            if user_response == 'insecticides':
                print("Chikoo: What about insecticides do you want information about? [Price, Type]")
                msg.body("Chikoo: What about insecticides do you want information about? [Price, Type]")
                SmsResponse.state = 1

            if(SmsResponse.greeting(request, user_response)is not None):
                print("Chikoo: " + SmsResponse.greeting(request, user_response))
                msg.body("Chikoo: " + SmsResponse.greeting(request, user_response))
            else:
                print("Chikoo: " + SmsResponse.response(user_response))
                msg.body("Chikoo: " + SmsResponse.response(user_response))
                sentence_tokens.remove(user_response)

        elif SmsResponse.state == 1:
            user_response = 'insecticides' + user_response
            SmsResponse.state = 0
        
        if user_response == 'bye':
            msg.body("Chikoo: Bye, take care! Let me know if you need help.")
            SmsResponse.state = -1

    return str(resp)



