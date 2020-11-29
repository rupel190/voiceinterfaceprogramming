
#import chatbot
from flask import Flask, render_template, request
import chatbot

server = Flask(__name__)


@server.route('/')
def hello():
    return render_template('index.html')
    
@server.route("/detectIntent", methods=['POST'])
def chat():
    answer = chatbot.response(request.get_data().decode('utf-8'))
    #if answer is None:
    #    return 'Did not get an answer. Great. Just great.'
    return answer
    
    #request.get_data()
    #return request.get_json
    

# FFS: needs to be at the bottom
if __name__ == '__main__':
    server.run(host='0.0.0.0')
    