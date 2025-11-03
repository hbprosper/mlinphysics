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
CONFIG = [
    'SRC_VOCAB_SIZE',
    'SRC_SEQ_LEN',
    'ENC_EMB_DIM', 
    'ENC_LAYERS', 
    'ENC_HEADS', 
    'ENC_FF_DIM', 
    'ENC_DROPOUT', 
    'TRG_VOCAB_SIZE',
    'TRG_SEQ_LEN',
    'DEC_EMB_DIM', 
    'DEC_LAYERS', 
    'DEC_HEADS', 
    'DEC_FF_DIM', 
    'DEC_DROPOUT', 
    'PAD_CODE',
    'SOS_CODE',
    'EOS_CODE'
]
def config_complete(config, keys=CONFIG):
    for key in keys:
        try:
            y = config(key)
        except:
            sys.exit(f'''❌
    Required configuration parameter, {key}, missing from configuration object!
    please check!
            ''')
    
def print_shape(a, x):
    print(f'{a:s}: {str(x.shape):s}')
    
# pretty print symbolic expression
def pprint(expr):
    try:
        display(sympify(expr))
    except:
        print(expr)

def stringify(codes, code2token):
    return ''.join([code2token[int(x)] for x in codes])
# ---------------------------------------------------------------
# build a simple tokenizer
# ---------------------------------------------------------------
# regular expression (regex) to extract tokens
get_tokens = re.compile('O[(]x[*][*]6[)]|[*][*]|[*]|[+]|[-]|[/]|'\
                        '[(]|[)]|[1-9][.]0|[0-9]|[a-zA-Z]+')

# split string "line" into tokens
def tokenize(line):
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
              replace('*', '[*]').\
              replace('-', '[-]').\
              replace('+', '[+]').\
              replace('(', '[(]').\
              replace(')', '[)]') 
                for _, x in tokens]

    cmd = r'^('+'|'.join(tokens)+')'
    cmd = re.compile(cmd)

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
            # this should never happen!
            
            print("problematic ***> ", line_orig)
            pprint(line_orig)
            raise ValueError(f'token not found:<<{line:s}>>')

    return tokens

# ---------------------------------------------------------------
# given a list of strings, build a token dictionary
# ---------------------------------------------------------------
def build_vocabulary(text, tokenize):
    
    tokens = set(['0','1','2','3','4','5','6','7','8','9'])
    for ii, line in enumerate(text):
        token_set = set(tokenize(line))
        tokens = tokens.union(token_set)

    tokens = list(tokens)
    tokens.sort()
    
    # ensure that PAD, SOS, and EOS symbols will always have codes 0, 1, 2
    tokens.insert(0, '<eos>') # end of sequence (EOS) symbol
    tokens.insert(0, '<sos>') # start of sequence (SOS) symbol
    tokens.insert(0, '<pad>') # padding (PAD) symbol   

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
class DataLoader:
    
    def __init__(self, filename, delimit, 
                 src_tokenize=tokenize,
                 trg_tokenize=tokenize,
                 max_seq_len=256, 
                 batch_size=64,
                 ftrain=18/20, # fraction of data devoted to training
                 fvalid=1/20,  # fraction of data devoted to validation
                 ftest=1/20,   # fraction of data devoted to testing              
                 device="cuda" if torch.cuda.is_available() else "cpu"):  
        
        max_seq_len -= 2

        # cache tokenizers
        
        self.src_tokenize = src_tokenize
        self.tgt_tokenize = trg_tokenize
        
        # cache computational device (CPU or GPU)
        
        self.device = device
        
        # read and split data into a list of 2-tuples
        
        print('read sequences')
        
        text = open(filename).readlines()
        data = [x.strip().split(delimit) for x in text]
        print(f'\n\tnumber of sequence pairs: {len(data)}\n')
        
        # print a few source/target pairs
        step = int(len(data)/3)
        for i, (src, trg) in enumerate(data):
            if i % step == 0:
                print(f'{i:6d} {"-"*83:s}')
                pprint(src)
                pprint(trg)
        print()
    
        # unzip into a list of sources and a list of targets
        
        srcs, tgts = zip(*data)

        # ---------------------------------------------------------------
        # build source vocabulary
        # ---------------------------------------------------------------      
        src_tokens, self.src_token2code, self.src_code2token = build_vocabulary(srcs, self.src_tokenize)
        print('source vocabulary')
        print(self.src_token2code)
        print()

        # ---------------------------------------------------------------
        # build target vocabulary
        # ---------------------------------------------------------------     
        tgt_tokens, self.tgt_token2code, self.tgt_code2token = build_vocabulary(tgts, self.tgt_tokenize)
        print('target vocabulary')
        print(self.tgt_token2code)
        print()
    
        # ---------------------------------------------------------------
        # tokenize sequences and map to integer codes
        # ---------------------------------------------------------------
        print('tokenize')
        
        # tokenize source sequences and map to integer codes
        
        srcs, avg_len, std_len, max_src_len = text2codes(srcs, self.src_token2code, self.src_tokenize)
        self.SRC_AVG_SEQ_LEN = avg_len
        self.SRC_STD_SEQ_LEN = std_len
        src_seq_len          = min(avg_len + 3 * std_len, max_src_len, max_seq_len)
        
        # tokenize target sequences and map to integer codes
        
        tgts, avg_len, std_len, max_tgt_len = text2codes(tgts, self.tgt_token2code, self.tgt_tokenize)
        self.TRG_AVG_SEQ_LEN = avg_len
        self.TRG_STD_SEQ_LEN = std_len
        tgt_seq_len          = min(avg_len + 3 * std_len, max_tgt_len, max_seq_len)

        # filter sequences
                
        self.srcs   = []
        self.tgts   = []
        for i, (src, tgt) in enumerate(zip(srcs, tgts)):
            if len(src) > src_seq_len: continue
            if len(tgt) > tgt_seq_len: continue
       
            self.srcs.append(src)
            self.tgts.append(tgt)
            
        # ---------------------------------------------------------------
        # split data into train, validation, and test sets
        # ---------------------------------------------------------------
        ftotal = ftrain + fvalid + ftest
        ftrain = ftrain / ftotal
        fvalid = fvalid / ftotal
        ftest  = ftest  / ftotal
        
        ntrain = int(len(self.srcs) * ftrain)
        nvalid = int(len(self.srcs) * fvalid)
        ntest  = int(len(self.srcs) * ftest)  
        
        # ---------------------------------------------------------------
        # cache data
        # ---------------------------------------------------------------
        self.train_data = [self.srcs[:ntrain], 
                           self.tgts[:ntrain]]

        self.valid_data = [self.srcs[ntrain:ntrain+nvalid], 
                           self.tgts[ntrain:ntrain+nvalid]]

        self.test_data  = [self.srcs[ntrain+nvalid:], 
                           self.tgts[ntrain+nvalid:]] 
        
        # ---------------------------------------------------------------
        # pad and delimit sequences. the codes for PAD, SOS, and EOS are
        # the same for source and target sequences
        # ---------------------------------------------------------------
        PAD = self.src_token2code['<pad>']
        SOS = self.src_token2code['<sos>']
        EOS = self.src_token2code['<eos>']

        self.PAD = PAD
        self.SOS = SOS
        self.EOS = EOS
        
        # pad training data
        
        print('delimit and pad training data')
        for i, (src, tgt) in enumerate(zip(self.train_data[0], 
                                           self.train_data[1])):
                
            self.train_data[0][i] = [SOS] + src \
              + (src_seq_len-len(src))*[PAD] + [EOS]
            
            self.train_data[1][i] = [SOS] + tgt \
              + (tgt_seq_len-len(tgt))*[PAD] + [EOS]
            
            if i % 1000 == 0:
                print(f'\r{i:6d}', end='')
        print()
        
        # pad validation data
        
        print('delimit and pad validation data')
        for i, (src, tgt) in enumerate(zip(self.valid_data[0], 
                                           self.valid_data[1])):

            self.valid_data[0][i] = [SOS] + src \
              + (src_seq_len-len(src))*[PAD] + [EOS]
            
            self.valid_data[1][i] = [SOS] + tgt \
              + (tgt_seq_len-len(tgt))*[PAD] + [EOS]
            
            if i % 1000 == 0:
                print(f'\r{i:6d}', end='')
        print()
        
        print('delimit test data but do not pad')
        
        for i, (src, tgt) in enumerate(zip(self.test_data[0], 
                                           self.test_data[1])):

            self.test_data[0][i] = [SOS] + src + [EOS]
            self.test_data[1][i] = [SOS] + tgt + [EOS]
            if i % 1000 == 0:
                print(f'\r{i:6d}', end='')
        print()

        self.SRC_SEQ_LEN     = len(self.train_data[0][0])
        self.SRC_VOCAB_SIZE  = len(self.src_token2code)
        
        self.TRG_SEQ_LEN     = len(self.train_data[1][0])
        self.TRG_VOCAB_SIZE  = len(self.tgt_token2code)

        print(f'avg. source sequence length: {self.SRC_AVG_SEQ_LEN:8d}')
        print(f'std. source sequence length: {self.SRC_STD_SEQ_LEN:8d}')
        print(f'     source sequence length: {self.SRC_SEQ_LEN:8d}')
        print(f'     source vocabulary size: {self.SRC_VOCAB_SIZE:8d}')
        print()
        
        print(f'avg. target sequence length: {self.TRG_AVG_SEQ_LEN:8d}')
        print(f'std. target sequence length: {self.TRG_STD_SEQ_LEN:8d}')
        print(f'     target sequence length: {self.TRG_SEQ_LEN:8d}')
        print(f'     target vocabulary size: {self.TRG_VOCAB_SIZE:8d}')
        print()

        # convert to tensors and load onto computational device
        # -------------------------------------------------------------
        self.train_x = torch.tensor(self.train_data[0]).to(self.device)
        self.train_t = torch.tensor(self.train_data[1]).to(self.device)
    
        self.valid_x = torch.tensor(self.valid_data[0]).to(self.device)
        self.valid_t = torch.tensor(self.valid_data[1]).to(self.device)

        self.test_x, self.test_t = self.test_data

        self.train_data = [self.train_x, self.train_t]
        self.valid_data = [self.valid_x, self.valid_t]
        
        print()
        print(f'training   data: '\
              f'{str(self.train_x.size()):s}, '\
              f'{str(self.train_t.size()):s}')
        
        print(f'validation data: '\
              f'{str(self.valid_x.size()):s}, '\
              f'{str(self.valid_t.size()):s}')
        
        print(f'test data:       '\
              f'{len(self.test_x):d}')
     
        self.batch_size = batch_size
        self.index      = 0 # for iterator
        
    # return a batch of data for the next step in the minimization
    def get_batch(self, data, ii=0, batch_size=None):
        x, t = data
        # selects at random "batch_size" integers from 
        # the range [0, batch_size-1] with replacement
        # corresponding to the row indices of the training 
        # data to be used
        if batch_size != None:
            rows = torch.randint(0, len(x)-1, size=(batch_size,))
        else:
            rows = torch.randint(0, len(x)-1, size=(self.batch_size,))

        # shape: [batch_size, seq_len]
        return x[rows], t[rows]
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        
    @property    
    def train_iterator(self):
        self.index       = 0
        self.data        = self.train_data
        self.num_batches = len(self.data[0]) // self.batch_size
        return self
    
    @property
    def valid_iterator(self):
        self.index       = 0
        self.data        = self.valid_data
        self.num_batches = len(self.data[0]) // self.batch_size
        return self
    
    @property
    def test_iterator(self):
        self.index       = 0
        self.data        = self.test_data
        self.num_batches = len(self.data[0])
        return self
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        return self

    def __next__(self):
        
        class Batch: pass
        
        if self.index < self.num_batches:
           
            batch = Batch()
            src, trg = self.data
           
            i = self.index * self.batch_size
            j = i + self.batch_size
                       
            if self.num_batches == len(src):
                batch.src = src[self.index].unsqueeze(0)
                batch.trg = trg[self.index].unsqueeze(0)
            else:
                batch.src = src[i:j]
                batch.trg = trg[i:j]
                        
            self.index += 1
            return batch
        else:
            self.index = 0
            raise StopIteration
    
    def data_splits(self):
        return self.train_data, self.valid_data, self.test_data

# ----------------------------------------------------------------
class SequenceData:
    
    def __init__(self, filename, 
                 max_seq_len=128,
                 delimit='|', 
                 src_tokenize=tokenize,
                 trg_tokenize=tokenize):  
        
        max_seq_len -= 2

        # cache tokenizers
        self.src_tokenize = src_tokenize
        self.trg_tokenize = trg_tokenize
        
        # ---------------------------------------------------------------      
        # read and split data into a list of 2-tuples
        # ---------------------------------------------------------------
        print('\treading sequences')
        try:
            text = open(filename).readlines()
        except:
            sys.exit(f'''❌
    Sequence data file {filename} NOT found!
            ''')
        print(f'\n\tsample size: {len(text)}\n')

        # unzip into a list of sources and a list of targets
        data = [x.strip().split(delimit) for x in text]
        for i, d in enumerate(data):
            if len(d) != 2:
                sys.exit(f'''❌
    Unable to split sequence at line {i} into source and target sequences
    using the delimiter {delimit}. Please check the file {filename}.
                ''')
        srcs, trgs = zip(*data)
        
        # print a few source/target pairs
        step = int(len(data)/1500)
        step = 500 * step
        for i in range(0, len(srcs), step):    
            print(f'{i}')
            pprint(srcs[i])
            pprint(trgs[i])
            print()

        # ---------------------------------------------------------------
        # build source vocabulary
        # ---------------------------------------------------------------  
        print('\tbuilding source vocabulary')
        src_tokens, self.src_token2code, self.src_code2token = build_vocabulary(srcs, self.src_tokenize)
        print(self.src_token2code)
        print()

        # ---------------------------------------------------------------
        # build target vocabulary
        # ---------------------------------------------------------------   
        print('\tbuilding target vocabulary')
        trg_tokens, self.trg_token2code, self.trg_code2token = build_vocabulary(trgs, self.trg_tokenize)
        print(self.trg_token2code)
        print()
    
        # ---------------------------------------------------------------
        # tokenize sequences and map to integer codes
        # ---------------------------------------------------------------
        
        # tokenize source sequences and map to integer codes
        
        print('\ttokenize sources')
        srcs, avg_len, std_len, max_src_len = text2codes(srcs, self.src_token2code, self.src_tokenize)
        self.SRC_AVG_SEQ_LEN = avg_len
        self.SRC_STD_SEQ_LEN = std_len
        src_seq_len          = min(avg_len + 3 * std_len, max_src_len, max_seq_len)
        
        # tokenize target sequences and map to integer codes
        
        print('\ttokenize targets')
        trgs, avg_len, std_len, max_trg_len = text2codes(trgs, self.trg_token2code, self.trg_tokenize)
        self.TRG_AVG_SEQ_LEN = avg_len
        self.TRG_STD_SEQ_LEN = std_len
        trg_seq_len          = min(avg_len + 3 * std_len, max_trg_len, max_seq_len)

        # filter sequences

        seqs = filter(lambda x: len(x[0]) <= src_seq_len and len(x[1]) <= trg_seq_len, zip(srcs, trgs))
        srcs, trgs = zip(*seqs)
        srcs = list(srcs) # need to be lists so that they are mutable
        trgs = list(trgs)
        
        # ---------------------------------------------------------------
        # pad and bracket sequences. the codes for PAD, SOS, and EOS are
        # the same for source and target sequences.
        # ---------------------------------------------------------------
        PAD = self.src_token2code['<pad>']
        SOS = self.src_token2code['<sos>']
        EOS = self.src_token2code['<eos>']

        self.PAD = PAD
        self.SOS = SOS
        self.EOS = EOS
        
        # pad and bracket data
        
        print('\tpad sequences and bracket with <sos> and <eos>')
        for i, (src, trg) in enumerate(zip(srcs, trgs)): 
            srcs[i] = [SOS] + src + (src_seq_len-len(src))*[PAD] + [EOS]
            trgs[i] = [SOS] + trg + (trg_seq_len-len(trg))*[PAD] + [EOS]

        # convert to numpy arrays and cache
        self.sources = np.array(srcs, dtype=np.int64)
        self.targets = np.array(trgs, dtype=np.int64)

        # print a summary
        self.SRC_SEQ_LEN     = len(srcs[0])
        self.SRC_VOCAB_SIZE  = len(self.src_token2code)
        
        self.TRG_SEQ_LEN     = len(trgs[0])
        self.TRG_VOCAB_SIZE  = len(self.trg_token2code)

        print()
        print('Summary')
        print(f' sample size: {len(self.sources)}')
        print(f'   source sequence length: {self.SRC_SEQ_LEN:8d}')
        print(f'   source vocabulary size: {self.SRC_VOCAB_SIZE:8d}')
        print()
        print(f'   target sequence length: {self.TRG_SEQ_LEN:8d}')
        print(f'   target vocabulary size: {self.TRG_VOCAB_SIZE:8d}')
        print()


