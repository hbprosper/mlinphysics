# Used in transformer tutorials
# ----------------------------------------------------------------
import os, sys, re
import numpy as np
import random as rn
import torch
import sympy as sp
try:
    from IPython.display import display
except:
    display = None

# symbols
from sympy import symbols, sympify, exp, \
    cos, sin, tan, \
    cosh, sinh, tanh, ln, log, E, O
x,a,b,c,d,f,g = symbols('x,a,b,c,d,f,g', real=True)
# ---------------------------------------------------------------
def print_shape(a, x):
    print(f'{a:s}: {str(x.shape):s}')
    
# Pretty print symbolic expression
def pprint(expr):
    try:
        display(sympify(expr))
    except:
        print(expr)

def stringify(codes, code2token):
    return ''.join([code2token[int(x)] for x in codes]).replace('<pad>', '')
# ---------------------------------------------------------------
# build a simple tokenizer
# ---------------------------------------------------------------
# regular expression (regex) to extract tokens
get_tokens = re.compile('O[(]x[*][*]6[)]|[*][*]|[*]|[+]|[-]|[/]|'\
                        '[(]|[)]|[1-9][.]0|[0-9]|[a-zA-Z]+')

# Split string "line" into tokens
def tokenize(line, lineno=-1):
    line_orig = line
    
    findall = get_tokens.findall
    # 1. get a unique list of tokens from string "line" and sort in
    #    decreasing length of token so that longest tokens, like "sinh",
    #    are searched before shorter tokens like "sin"
    tokens  = [(len(x), x) for x in list(set(findall(line)))]
    tokens.sort()
    tokens.reverse()
    
    # 2. create a regex to search for any token in the list of tokens
    #    (make sure that regex special symbols are not used as such)
    tokens = [x.\
              replace('/', '[/]').\
              replace('*', '[*]').\
              replace('-', '[-]').\
              replace('+', '[+]').\
              replace('(', '[(]').\
              replace(')', '[)]') 
                for _, x in tokens]

    cmd_str = r'^('+'|'.join(tokens)+')'
    cmd = re.compile(cmd_str)

    # 3. loop through string and match a token starting at the
    #    1st character of the string. then shorten the string
    #    by removing the matched token and repeat until the
    #    string as zero length.
    max_len = len(line)
    tokens  = []
    j = 0
    while (len(line) > 0) and j < max_len:
        j += 1

        token = cmd.findall(line)

        if len(token) > 0:
            tokens.append(token[0])
            line = cmd.sub('', line)
        else:
            # This should never happen!
            print(f"Problematic sequence {lineno}: ")
            pprint(line_orig)
            raise ValueError('No token found!')
            
    return tokens
# ---------------------------------------------------------------
# Given a list of strings, build a token dictionary
# ---------------------------------------------------------------
def build_vocabulary(text, tokenize):
    
    tokens = set(['0','1','2','3','4','5','6','7','8','9'])
    for ii, line in enumerate(text):
        token_set = set(tokenize(line, lineno=ii))
        tokens = tokens.union(token_set)

    tokens = list(tokens)
    tokens.sort()
    
    # IMPORTANT: PAD, SOS, EOS, and SEP symbols will always have codes 0, 1, 2, 3
    tokens.insert(0, ' ')     # space
    tokens.insert(0, '<sep>') # separator
    tokens.insert(0, '<eos>') # end of sequence symbol 
    tokens.insert(0, '<sos>') # start of sequence symbol
    tokens.insert(0, '<pad>') # padding symbol

    # token to code map (it seems that we need to start from code 0)
    codes      = np.arange(len(tokens)).tolist()
    token2code = dict(zip(tokens, codes))
    code2token = dict(zip(codes, tokens))
    
    return tokens, token2code, code2token
# ---------------------------------------------------------------
# map tokens to codes
# ---------------------------------------------------------------
def text2codes(text, token2code, tokenize, step=2000):
    
    max_len = 0    # maximum length of token sequences
    avg_len = 0.0  # sum len_i
    std_len = 0.0  # sum len_i**2
    
    codes   = []   # tokenized string mapped to integer codes

    for i, line in enumerate(text):

        # map source tokens to integer codes
        cds = [token2code[t] for t in tokenize(line)]
        codes.append(cds)

        # get maximum string length (in tokens)
        l   = len(cds)
        if l > max_len:
            max_len = l

        avg_len += l
        std_len += l * l

        # i'm alive printout!
        if i % step == 0:
            print(f'\r{i:6d}', end='')

    print()

    # compute average and standard deviation

    avg_len /= len(text)
    std_len /= len(text)
    std_len  = np.sqrt(std_len - avg_len**2)
    
    avg_len  = int(avg_len+0.5)
    std_len  = int(std_len+0.5)
    
    return codes, avg_len, std_len, max_len
# ----------------------------------------------------------------
class SequenceData:
    
    def __init__(self, filename, 
                 max_seq_len=128,
                 delimit='|', 
                 tokenize=tokenize):  

        self.filename = filename
        max_seq_len -= 2

        # cache tokenizer
        self.tokenize = tokenize
        # --------------------------------------------------------------- 
        # Read sequence data 
        # ---------------------------------------------------------------
        print('\treading prompt/target sequences')
        try:
            text = [x.strip() for x in open(filename).readlines()]
        except:
            raise RuntimeError(f'''❌
    Sequence data file {filename} NOT found!
            ''')
        print(f'\n\tsample size: {len(text)}\n')

        # ---------------------------------------------------------------
        # Split into prompts and targets
        # ---------------------------------------------------------------
        prompts, targets = zip(*[x.split(delimit) for x in text])

        # Print a few prompt/target pairs
        nseq = len(text)
        step = int(nseq/1500)
        step = 500 * step
        for i in range(0, nseq, step):    
            print(f'{i}')
            pprint(prompts[i])
            pprint(targets[i])
            print()
        # ---------------------------------------------------------------
        # Build vocabulary
        # ---------------------------------------------------------------  
        print('build vocabulary')
        tokens, self.token2code, self.code2token = build_vocabulary(
            prompts + targets, self.tokenize)
        print(self.token2code)
        print()
        
        self.VOCAB_SIZE  = len(self.token2code)
        PAD = self.token2code['<pad>']
        SOS = self.token2code['<sos>']
        EOS = self.token2code['<eos>']
        SEP = self.token2code['<sep>']
        
        self.PAD = PAD
        self.SOS = SOS
        self.EOS = EOS
        self.SEP = SEP
        # ---------------------------------------------------------------
        # Tokenize sequences and map to integer codes
        # ---------------------------------------------------------------        
        print('tokenize prompts and targets')
        prompts, _,_,_ = text2codes(
            prompts, self.token2code, self.tokenize)
        targets, _,_,_ = text2codes(
            targets, self.token2code, self.tokenize)
        
        # Now concatenate prompts and targets
        print('concatenate prompts and targets')
        self.sequences = [p + [SEP] + t for p, t in zip(prompts, targets)]
        
        # Filter sequences by sequence length.
        self.sequences = list(filter(
            lambda x: len(x) <= max_seq_len, self.sequences))
        
        # ---------------------------------------------------------------
        # Pad and bracket sequences. 
        # ---------------------------------------------------------------
        print('pad sequences and bracket with <sos> and <eos>')
        avg_seq_len = 0.0
        std_seq_len = 0.0
        for i, sequence in enumerate(self.sequences): 
            seq_len = len(sequence)
            avg_seq_len += seq_len
            std_seq_len += seq_len**2
            
            padding = (max_seq_len-seq_len)*[PAD] 
            
            self.sequences[i] = [SOS] + sequence + padding + [EOS]
            
        self.sequences = np.array(self.sequences)
        
        avg_seq_len /= len(self.sequences)
        std_seq_len /= len(self.sequences)
        std_seq_len  = np.sqrt(std_seq_len - avg_seq_len**2)
        
        print()
        print('Summary')
        print(f' sample size:                {len(self.sequences):8d}')
        print(f'  avg(sequence length):      {avg_seq_len:8.1f}')
        print(f'  stdv(sequence length):     {std_seq_len:8.1f}')
        print(f' vocabulary size:            {self.VOCAB_SIZE:8d}')
        print()

    def split(self, sequence):
        j = list(sequence.flatten()).index(self.SEP)
        prompt = sequence[0, :j+1].view(1, -1)   # Delimited prompt, shape: [1, prompt_len]
        target = sequence[0, j+1:-1]             # Un-delimited target, shape: [target_len]  
        return prompt, target

    def str(self, sequence):
        return stringify(sequence, self.code2token)
        
    def pprint(self, sequence):
        '''
    Print a single sequence.
    
    Arguments:
        sequence: numpy array or a tensor
        '''
        # Determine if we have a 2D shape
        try:
            if sequence.ndim == 2:  # shape
                sequence = sequence[0]
        except:
            # sequence probably doesn't have an ndim attribute
            pass
            
        print('Sequence')
        print(sequence)
        print()

        sequence = sequence[1:-1]          # Strip away <sos> and <eos> tokens
            
        j = list(sequence).index(self.SEP) # Find index of separator token
        
        prompt = self.str(sequence[:j])
        print('Prompt')
        pprint(prompt)
        
        print('Target')
        target = self.str(sequence[j+1:])
        pprint(target)        