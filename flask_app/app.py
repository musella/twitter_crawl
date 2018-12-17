from flask import Flask, render_template, request, redirect, jsonify, url_for
import random
import string
import logging
import json
# import httplib
import requests
from flask import make_response
import pickle

import sys

sys.path.append("../python")

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from copy import copy

from models import BeamSearch

app = Flask(__name__)
model = None
tk = None
nsuggestions = 5
padto = None
data_folder = "../data/v1"
gen_model = data_folder+"/models/gru_unk/gru-model.hd5"
## data_folder = "../data/v2"
## gen_model = data_folder+"/models/gru/gru-model.hd5"
tok_model = data_folder+"/models/sequences/tokenizer.pkl"

beam_search = BeamSearch(gen_model,tok_model)

def load_my_model():
        global model
        global padto
        global tk
        model = load_model(gen_model,compile=False)
        padto = int(model.input.shape[1])
        
        with open(tok_model,"rb") as fin:
                tk = pickle.loads(fin.read())
        print('Model '+str(model))
        print('Tokenizer '+str(tk))
                
def run_beam_search(request,horizon):
        if request.method == 'POST':
                print("POST request")
        else:
                print("GET request")
                
        text = request.json['input']

        print("Horizon", horizon)
        print("Text", text)
        suggestions = beam_search.predict(text,nsuggestions,horizon)
                
        return jsonify(suggestions=suggestions)

        
@app.route('/autocomplete', methods=['GET', 'POST'])
def autocomplete():
        what = request.json["what"]
        if what == "get_next_word":
                return run_beam_search(request,1)
        elif what == 'get_sentence':
                return run_beam_search(request,50)
        

# Route to home page
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def show_home():
        if beam_search.model is None:
                beam_search.load_model()
        return render_template('index.html')


# Main Method
if __name__ == '__main__':
        app.secret_key = 'super_secret_key'
        app.debug = False
        load_my_model()
        app.run(host='0.0.0.0', port=5000)
