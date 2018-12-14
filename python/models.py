from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import numpy as np

import pickle

# ----------------------------------------------------------------------------------------------
class BeamSearch:
    """ Completes words sequences using the beam search algorithm
    Beam search is a breadth-first search algorithm that expands only the 
    top-n branches. 
    
    Here the branches are ranked by the (log) probabilities predicted by the 
    sequence generation model.
    
    https://en.wikipedia.org/wiki/Beam_search
    """

    # ------------------------------------------------------------------------------------------
    def __init__(self,gen_model,tok_model):
        """ Constructor 
        
        Arguments: 
        gen_model -- path to to generative model weights
        tok_model -- path to the string tokenizer
        """
        
        self.gen_model = gen_model
        self.tok_model = tok_model

        self.model = None
        self.tk = None

    # ------------------------------------------------------------------------------------------
    def load_model(self):
        """ Load generative model and tokenizer """

        self.model = load_model(self.gen_model,compile=False)
        self.padto = int(self.model.input.shape[1])
        
        with open(self.tok_model,"rb") as fin:
                self.tk = pickle.loads(fin.read())

        self.stop_word = self.tk.word_index.get("<stop>",None)
        self.unk_word =  self.tk.word_index.get("<unk>",None)
        
        print('Model '+str(self.model))
        print('Tokenizer '+str(self.tk))

        
    # ------------------------------------------------------------------------------------------
    def predict_next(self,seqs,log_probs0,nsuggestions):
        """ Expand the next search level and return candidate sequences
        
        Arguments:
        seqs        -- set of sequences currently in the beam
        log_probs0  -- log-probabilities associated to the current sequences
        nsuggetions -- number of suggested sequences to return (= beam width)

        Returns 
        candidates -- list of most probabile nsuggetions sequences
        log_probs1 -- associated log-probabilities 

        """

        # check which sequences have already a stop word
        candidates = []
        log_probs1 = []
        seqs_to_extend = []
        for seq,log_prob0 in zip(seqs,log_probs0):
            if seq[-1] == self.stop_word:
                candidates.append(seq)
                log_probs1.append(log_prob0)
            else:
                seqs_to_extend.append(seq)

        # run the model prediction for sequences that have not yet ended
        X = pad_sequences(seqs_to_extend,self.padto)
        probs = self.model.predict(X)[:,-1,:]
        if self.unk_word is not None:
            probs[:,self.unk_word] = 0.

        # extract the top nsuggestions for each sequence that needs to be expanded
        preds = np.flip(probs.argsort(-1)[:,-nsuggestions:],-1)
        # expend the current sequences with the new suggestions and compute the
        # corresponding log-probabilities
        for seq, log_prob0, pred, prob in zip(seqs,log_probs0,preds,probs):
            seq = seq
            for ipred in pred:
                ilog_prob = np.log(prob[ipred])
                candidates.append( seq+[ipred] )
                log_probs1.append( log_prob0+ilog_prob )

        # extract and return the nsuggestions with largest log-probabilities
        log_probs1 = np.array(log_probs1)
        keep = np.flip( log_probs1.argsort(-1)[-nsuggestions:], -1 )
        return [ candidates[icand] for icand in keep], log_probs1[keep] 
        
        
    # ------------------------------------------------------------------------------------------
    def predict(self,stub,nsuggestions,horizon):
        """ Expand a word sequences, given its initial stub.

        The seach is run using a beam witdh of nsuggestions and until horizon 
        levels have been expended.
        
        Returns: the nsuggestions most probable completion
        """
        
        if self.model is None:
            self.load_model()
            
        seq = self.tk.texts_to_sequences( [stub,stub] )
        print(seq)
        nel = len(seq[0])
        
        candidates = seq
        log_probs = [0.]
        
        for istep in range(horizon):
            candidates, log_probs = self.predict_next( candidates, log_probs, nsuggestions )
        candidates = self.tk.sequences_to_texts( [cand[nel:] for cand in candidates] )
        
        return [stub+" "+cand for cand in candidates]
