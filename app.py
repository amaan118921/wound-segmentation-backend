from flask import Flask, request
import predict as pr

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'hello, world!'


@app.route('/predict')
def predict():
    # url = request.args.get('url')
    # filename = request.args.get('filename')
    url = 'https://firebasestorage.googleapis.com/v0/b/womensafety-c4d41.appspot.com/o/foot-ulcer-0027.png?alt=media&token=01577279-05f7-48b3-8e8d-52e318ea3cec'
    name = "resultttt.jpg"
    if pr.predict_result(url, name): return 'success'
    return 'failed'
