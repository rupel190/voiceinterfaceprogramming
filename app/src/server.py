 
#import chatbot
from flask import Flask, render_template, request
import chatbot

server = Flask(__name__)


@server.route('/')
def hello():
    return render_template('index.html')
    
@server.route("/detectIntent", methods=['POST'])
def chat():

    # variables used in response should be held here but can't be quickly changed
    sentence = request.get_data().decode('utf-8')
    answer = chatbot.response()
    
    #if answer is None:
    #    return 'Did not get an answer.
    return answer
    
    #request.get_data()
    #return request.get_json
    

# FFS: needs to be at the bottom of course
if __name__ == '__main__':
    server.run(host='0.0.0.0')
    
    
    
    
    
