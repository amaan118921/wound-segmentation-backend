from flask import Flask, request
import predict as pr

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'hello, world!'


@app.route('/predict')
def predict():
    url = request.args.get('url')
    filename = request.args.get('filename')
    name = filename + '.jpg'
    if pr.predict_result(url, name): return 'success'
    return 'failed'
