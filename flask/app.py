from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = "tempSecret"
socketio = SocketIO(app, cors_allowed_origins="*")

'''
@app.route('/')
def hello_world():
    return "Hello World, from the Gesture Recognizer Server!"

@app.route('/predict')
def predict():
    pass
'''
if __name__ == "__main__":
    socketio.run(app)
    # app.run(host="0.0.0.0", port=5000)