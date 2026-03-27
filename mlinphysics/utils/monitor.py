#------------------------------------------------------------------------------
# Real time monitoring of loss curves during training
# Harrison B. Prosper
# July 2021
#------------------------------------------------------------------------------
import os, sys, re
import numpy as np
try:
    import pandas as pd
except:
    raise ImportError('''
    Please install pandas:

        conda install pandas
        
    ''')
import time
try:
    import matplotlib as mp
except:
    raise ImportError('''
    Please install matplotlib:

        conda install matplotlib
    ''')
#------------------------------------------------------------------------------
DELAY = 5 # seconds - interval between plot updates
LOG_SWITCH = 3
CHECK = "\u2705"
FAIL  = "\u274C"
WARN  = "\u26A0"
#------------------------------------------------------------------------------
# The loss file should be a simple text file with olumns of numbers:
#
#   iterations,train-losses,validation-losses,...
#     
def get_losses(loss_file):
    try:
        losses = pd.read_csv(loss_file).to_numpy()
        return losses[:, 0], losses[:, 1], losses[:, 2]
    except:
        return None

def get_timeleft(timeleft_file):
    if timeleft_file == None:
        return None
    try:
        return open(timeleft_file, 'r').read().strip()
    except:
        return None

class TimeLeft:
    '''
    Return the amount of time left.
    
    timeleft = TimeLeft(N)
    
    N: maximum loop count
    
      for i in timeleft:
          : :

    or
       timeleft(i, extra)
      
    '''
    def __init__(self, N):
        self.N = N        
        self.timenow = time.time
        self.start = self.timenow()
        self.str = ''
        
    def __del__(self):
        pass
    
    def __timestr(self, ii):
        # elapsed time since start
        elapsed = self.timenow() - self.start
        s = elapsed
        h = int(s / 3600) 
        s = s - 3600*h
        m = int(s / 60)
        s = s - 60*m
        hh= h
        mm= m
        ss= s
        
        # time/loop
        count = ii+1
        t = elapsed / count
        f = 1/t if count > 10 else 0.0
        
        # time left
        s = t * (self.N - count)
        h = int(s / 3600) 
        s = s - 3600*h
        m = int(s / 60)
        s =  s - 60*m
        percent = 100 * count / self.N

        return "%10d|%6.2f%s|%2.2d:%2.2d:%2.2d|%2.2d:%2.2d:%2.2d|%6.1f it/s" % \
            (ii, percent, '%', hh, mm, ss, h, m, s, f)
        
    def __iter__(self):
        
        for ii in range(self.N):
            
            if ii < self.N-1:
                print(f'\r{self.__timestr(ii):s}', end='')
            else: 
                print(f'\r{self.__timestr(ii):s}')
                
            yield ii
            
    def __call__(self, ii, extra='', colorize=False):
        
        if extra != '':
            if colorize:
               extra = "\x1b[1;34;48m|%s\x1b[0m" % extra
                
        self.a_str = f'{self.__timestr(ii):s}{extra:s}'
        
        if ii < self.N-1:
            print(f'\r{self.a_str}', end='')
        else:
            print(f'\r{self.a_str}')

    def __str__(self):
        return self.a_str
#--------------------------------------------------------------------
class LossMonitor:    
    '''    
    monitor = LossMonitor(lossfile, [ylabel=R(omega), ylog=True, xlog=False])
        :   :
    monitor()
    '''
    def __init__(self, lossfile, 
                 ylabel='$R(\\omega)$', 
                 ylog=True, 
                 xlog=False):

        self.lossfile = lossfile
        self.timeleftfile = lossfile.replace('.csv', '.txt')
        self.ylabel = ylabel
        self.ylog = ylog
        self.xlog = xlog
        
        # get first blocking backend
        self.original_backend = mp.get_backend()
        for backend in ("TkAgg", "QtAgg", "MacOSX"):
            try:
                mp.use(backend, force=True)
                break
            except:
                backend = None
        if backend == None:
            print(f'{WARN} No suitable GUI (blocking) backend found for Monitor!')
        else:
            print(f'\nplotting backend: {backend}')
        import matplotlib.pyplot as plt

        # set up an empty figure
        self.fig = plt.figure(figsize=(8, 4))
        self.fig.suptitle(self.lossfile)

        # add a subplot to it
        nrows, ncols, index = 1,1,1
        self.ax = self.fig.add_subplot(nrows, ncols, index)

    def __update(self, frame=None):            
        fig, ax = self.fig, self.ax
        
        ax.clear()
        ax.set_xlabel('Iteration', fontsize=14)
        ax.set_ylabel(self.ylabel, fontsize=14)
        ax.grid(True, which="both", linestyle='-')
        
        data = get_losses(self.lossfile)
        if type(data) != type(None):
            
            iters, train, valid = data
            
            if len(train) > 0:

                if self.ylog:
                    if train[0]/train[-1] > LOG_SWITCH:
                        ax.set_yscale('log')
                    
                if self.xlog:
                    if len(iters) > 10:
                        ax.set_xscale('log')
                    
                timeleft = get_timeleft(self.timeleftfile)
                if timeleft != None:
                    ax.set_title(timeleft, fontsize=11)
                else:
                    ax.set_title('Iteration: %5d|%s' % (iters[-1], time.ctime()))
                    
                ax.plot(iters, train, c='red',  linestyle='dashed', label='training')
                ax.plot(iters, valid, c='blue', label='validation')
                ax.legend()
                
        fig.tight_layout()
        
    def show(self):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        self.anim = FuncAnimation(
            fig=self.fig, 
            func=self.__update, 
            interval=1000*DELAY, # milliseconds
            repeat=False, 
            cache_frame_data=False)

        # this should block
        plt.show()
        print('\nciao!')
#--------------------------------------------------------------------
class Monitor:
    '''
    Write training and validation losses to a csv file and optionally
    the model parameters. The losses can be monitored while training 
    by running the command

        monlosses losses.csv&

    where losses.csv is the name of the loss file
    '''

    def __init__(self, 
                 niterations, 
                 lossfile,  
                 monitorstep,
                 newlossfile=True,
                 frac=0.005,
                 model=None, 
                 paramsfile=None, 
                 ylabel=None):
      
        # cache inputs
        self.niterations = niterations
        self.lossfile = lossfile
        self.monitorstep = monitorstep
        self.newlossfile = newlossfile
        self.frac = frac
        self.model = model
        self.paramsfile = paramsfile
        self.timeleftfile = lossfile.replace('.csv', '.txt')       
        self.minavloss = float('inf')  # initialize minimum average loss
        self.ylabel = ylabel

        # In case the graphics backend changes, let's 
        # cache current backend and restore in the end function
        self.original_backend = mp.get_backend()
    
        # initialize loss file
        # create loss file if it does not exist
        if not os.path.exists(lossfile) or newlossfile:
            open(lossfile, 'w').write('iteration,train,val,valbest,lr\n')  
    
        self.reset()
        
    def __call__(self, t_loss, v_loss, lr=0, epoch=None):
        
        loss_decreased = v_loss < (1 - self.frac) * self.minavloss
        if loss_decreased:
            self.min_avloss = v_loss
        v_best_loss = self.minavloss
        
        # update loss file
        jj = self.ii-1 # the update occurs in  step()
        self.itno = self.offset + jj

        open(self.lossfile, 
             'a').write(f'{self.itno:10d},'
                        f'{t_loss:9.3e},{v_loss:9.3e},{v_best_loss:9.3e},{lr:9.3e}\n')

        # if specified save model parameters
        if type(self.model) != type(None):
            if loss_decreased:
                self.model.save(self.paramsfile)

        # update time left file
        if epoch != None:
            line = f'|{t_loss:9.3e}|{v_loss:9.3e}|{epoch:10d}|'
        else:
            line = f'|{t_loss:9.3e}|{v_loss:9.3e}|{self.itno:10d}|'
            
        self.timeleft(jj, line)
        open(self.timeleftfile, 'w').write(f'{str(self.timeleft):s}\n')

    def step(self):
        save = self.ii % self.monitorstep == 0
        self.ii += 1
        return save

    def reset(self):
        # get last iteration number from loss file
        df = pd.read_csv(self.lossfile)

        # initialize iteration counters: absolute and relative
        if len(df) < 1:
            self.offset = 0
        else:
            self.offset = df.iteration.iloc[-1] # get last iteration number

        # relative counter
        self.ii = 0
        
        self.timeleft = TimeLeft(self.niterations)

    def start(self):
        import subprocess

        self.reset()
        
        cmd = ["monlosses", self.lossfile]
        if self.ylabel != None:
            cmd.append(self.ylabel)
        print(' '.join(cmd))
        
        self.p = subprocess.Popen(cmd,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
    def end(self):
        mp.use(self.original_backend, force=True)