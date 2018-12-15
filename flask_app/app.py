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
nsuggestions = 3
horizon = 1
## span = 3 
padto = None
## data_folder = "../data/v1"
data_folder = "../data/v1"
gen_model = data_folder+"/models/gru_unk/gru-model.hd5"
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
                

@app.route('/datcomplete', methods=['GET', 'POST'])
def datcomplete():
        print("** datcomplete() called")
        if request.method == 'POST':
                print("POST request")
        else:
                print("GET request")


        ### if model is None:
        ###         load_my_model()
                
        jsonData = request.json['variable']

        suggestions = beam_search.predict(jsonData,nsuggestions,1)
        ### for horizon in range(1,10):
        ###         suggestions += beam_search.predict(jsonData,nsuggestions,horizon)
        ###         suggestions += ["-"*50]
                
        return jsonify(suggestions=suggestions)
        
        ## seq = tk.texts_to_sequences( [jsonData] )
        ## nel = len(seq[0])
        ## X = pad_sequences(seq,padto)
        ## preds = model.predict(X)[:,-1,:].argsort(-1)[0,-nsuggestions:]
        ## preds = tk.sequences_to_texts( [ [pred] for pred in reversed(preds) ] )
        ## suggestions = [ jsonData+" "+pred for pred in preds ]
        ## return jsonify(suggestions=suggestions)
        

# Route to home page
@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def show_home():
        return render_template('index.html')


# Main Method
if __name__ == '__main__':
        app.secret_key = 'super_secret_key'
        app.debug = False
        load_my_model()
        app.run(host='0.0.0.0', port=5000)
