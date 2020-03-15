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
chatbots_file = open('chatbot.txt','r', errors = 'ignore')
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
        return None



#Allow user to exit out from bot responses, and add addtional question clarifications.
app = Flask(__name__)

@app.route('/bot', methods=['POST'])
def bot():
    
    user_response = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()
    responded = False

    if user_response is None:
        msg.body("Sorry, I did not hear a response.")

    elif SmsResponse.state == -1:
        msg.body("Chikoo deactivated already.")
    else:
        if SmsResponse.state == 0:
            if user_response == 'Crop Prices':
                msg.body("Chikoo: What commodity do you want the current market information about?")
                SmsResponse.state = 1
            if 'fertilizer' in user_response:
                msg.body("Chikoo: Where are you located? (City, Country)")
                SmsResponse.state = 1
            
            possible_greet = SmsResponse.greeting(request, user_response)
            if(possible_greet is not None):
                msg.body("Chikoo: " + SmsResponse.greeting(request, user_response) + " My name is Chikoo. Ask me about the market, and all your agriculture needs.")
            else:
                msg.body("Chikoo: " + SmsResponse.response(user_response))
                sentence_tokens.remove(user_response)

        elif SmsResponse.state == 1:
            if user_response == 'Crop Prices':
                user_response = 'Crop Prices' + user_response
                SmsResponse.state = 0
            if 'fertilizer' in user_response:
                user_response = 'Fertilizer Use in ' + user_response
        
        if user_response == 'bye':
            msg.body("Chikoo: Bye, take care! Let me know if you need help.")
            SmsResponse.state = -1


        

    return str(resp)
