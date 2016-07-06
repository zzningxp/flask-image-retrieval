import simplejson as json
from flask import render_template, request, make_response, send_file, redirect, url_for, Markup
import flask
import sys
from shutil import rmtree
import os
import time
from hashlib import sha1

import image_prehandler
import image_caffe
import image_retrieval

DATA_DIR = 'data'
asyncresult = {}

app = flask.Flask(__name__, static_folder=DATA_DIR)
app._static_folder = os.path.abspath(DATA_DIR)
app.debug = False  # TODO: make sure this is False in production
#app.debug = True

try:  
# Reset saved files on each start
#    rmtree(DATA_DIR, True)
    os.mkdir(DATA_DIR)
except OSError:
    pass

@app.route('/')
def index():
    return redirect(url_for('search_by_image'))

@app.route('/search_by_image')
def search_by_image():
    return render_template('index.html')

@app.route('/stream/<imageid>')
def stream(imageid):
    return flask.Response(event_stream(flask.request.access_route[0], imageid),
                          mimetype='text/event-stream')

def event_stream(client, imageid):
    for message in receive(imageid):
        print 'message:', message
        yield 'data: {0}\n\n'.format(message)

def receive(imageid):
    try:
        ret = asyncresult[imageid]
        del asyncresult[imageid]
        yield ret
    except:
        pass
    yield ''

@app.route('/post', methods=['POST'])
def post():
    sha1sum = sha1(flask.request.data + str(time.time())).hexdigest()
    target = os.path.join(DATA_DIR, '{0}.jpg'.format(sha1sum))
    try:
        if image_prehandler.save_normalized_image(target, flask.request.data):
            if image_caffe.feature_exact(target):
                list = image_retrieval.retrieval(target)
            message = {'src': target, 'ip_addr': safe_addr(flask.request.access_route[0]), 'resultsize': len(list)}
            for i,img in enumerate(list):
                message['result%s'%i] = img
            asyncresult[sha1sum] = json.dumps(message)
    except Exception as e:  # Output errors
        print e
        return '{0}'.format(e)

    return sha1sum

def safe_addr(ip_addr):
    """Strip of the trailing two octets of the IP address."""
    return '.'.join(ip_addr.split('.')[:2] + ['xxx', 'xxx'])

if __name__ == '__main__':
    app.run(threaded=True)
