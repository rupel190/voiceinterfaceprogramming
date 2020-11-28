
#import chatbot
from flask import Flask, render_template, request
import chatbot

server = Flask(__name__)


@server.route('/')
def hello():
    return render_template('index.html')

#@server.route('/chat/<message>')
#def robot(): 
    #global response
    #return render_template('chat.html', message, response(message))

    

    
@server.route("/detectIntent", methods=['POST'])
def chat():
    # unescaped as is
    answer = chatbot.response('Hey')
    if answer is None:
        return 'Did not get an answer. Great. Just great.'
    return answer
    
    #request.get_data()

    # unescaped as json
    #return request.get_json
    
    
    #return render_template('chat.html', message, response(message))
    #global BOT
    #return render_template('index.html', noteresponse=print_response(BOT, request.form["chat"]))
    
    
# FFS: needs to be at the bottom
if __name__ == '__main__':
    server.run(host='0.0.0.0')
    