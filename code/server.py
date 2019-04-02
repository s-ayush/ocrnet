import BaseHTTPServer
import json
from network2 import *
import numpy as np

HOST_NAME = 'localhost'
PORT_NUMBER = 8000
HIDDEN_NODE_COUNT = 100
EPOCH = 30
MINI_BATCH_SIZE = 10
REGULARIZATION = 5.0
LEARNING_RATE = 0.1

# Setup for Nielsen's network
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network([784, HIDDEN_NODE_COUNT, 10], cost=CrossEntropyCost)
net.large_weight_initializer()

net.SGD(training_data, EPOCH, MINI_BATCH_SIZE, LEARNING_RATE, evaluation_data=test_data, lmbda=REGULARIZATION, monitor_evaluation_accuracy=True)

class JSONHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_POST(s):
        response_code = 200
        response = ""
        var_len = int(s.headers.get('Content-Length'))
        content = s.rfile.read(var_len);
        payload = json.loads(content);

        if payload.get('predict'):
            try:
                response = {"type":"test", "result":net.predict(payload['image'], True)}
            except:
                response_code = 500
        else:
            response_code = 400

        s.send_response(response_code)
        s.send_header("Content-type", "application/json")
        s.send_header("Access-Control-Allow-Origin", "*")
        s.end_headers()
        if response:
            s.wfile.write(json.dumps(response))
        return

if __name__ == '__main__':
    server_class = BaseHTTPServer.HTTPServer;
    httpd = server_class((HOST_NAME, PORT_NUMBER), JSONHandler)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    else:
        print "Unexpected server exception occurred."
    finally:
        httpd.server_close()
