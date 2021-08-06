from flask import Flask
from flask_socketio import SocketIO
import cv2
import io
from PIL import Image
import base64
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = "tempSecret"
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('image')
def image(data_image):
    print("received image")
    sbuf = io.StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    cv2.imshow('mat', frame)
    cv2.waitKey(0)

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