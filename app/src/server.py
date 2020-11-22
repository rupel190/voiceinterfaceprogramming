
import chatbot
from flask import Flask, render_template


server = Flask(__name__)

@server.route('/')
def hello():
    return render_template('index.html')

@server.route('/chat/<message>')
def robot(message):
    return render_template('chat.html', message=message)
    
# @server.route("/<string:message>")
@server.route("/response", methods=['GET','POST'])
def reponse():
    return render_template('index.html')
    #global BOT
    #return render_template('index.html', noteresponse=print_response(BOT, request.form["chat"]))
    
    
# FFS: needs to be at the bottom
if __name__ == '__main__':
    server.run(host='0.0.0.0')
    