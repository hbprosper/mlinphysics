#-------------------------------------------------------------------------
# Real time monitoring of loss curves during training
# Harrison B. Prosper
# Created: July 2021
# Updated: Jun 12 2026 - Use tensorboard by default, if installed.
# Updated: Aug 21 2026 - Use standalone monitor by defualt!
#-------------------------------------------------------------------------
import os, sys, re
import numpy as np
from pathlib import Path
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except:
    TENSORBOARD_AVAILABLE = False
    print('''
    tensorboard not installed, will default to builtin loss monitor.
    ''')
#-------------------------------------------------------------------------
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
#-------------------------------------------------------------------------
DELAY = 5 # seconds - interval between plot updates
LOG_SWITCH = 3
CHECK = "\u2705"
FAIL  = "\u274C"
WARN  = "\u26A0"
#-------------------------------------------------------------------------
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
        return self.a_str

    def __str__(self):
        return self.a_str()
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
    the model parameters. By default, the losses are displayed by a 
    simple standalone display is activated in a separate window.
    However, if tensorboard is available, and desired, set
    use_tensorboard=True.
    '''

    def __init__(self, 
                 niterations, 
                 lossfile,  
                 monitorstep,
                 newlossfile=True,
                 frac=0.005,
                 model=None, 
                 paramsfile=None,
                 use_tensorboard=False,
                 ylabel=None):
      
        # cache inputs
        self.niterations = niterations
        self.lossfile    = lossfile
        self.monitorstep = monitorstep
        self.newlossfile = newlossfile
        self.frac  = frac
        self.model = model
        self.paramsfile  = paramsfile
        
        p = Path(lossfile)
        self.timeleftfile= lossfile.replace('.csv', '.txt')
        self.checkfile = lossfile.replace(f'{p.name}', 'checkpoint.csv')
        
        self.minavloss = float('inf')  # initialize minimum average loss
        self.ylabel = ylabel

        # get absolute path to runs folder
        # runs_pathname = p.parent.resolve()

        use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE

        if use_tensorboard:
            # create a tensorboard writer if tensorboard is installed
            self.writer = SummaryWriter(log_dir=os.path.expanduser('~/runs'))
        else:
            self.writer = None
            
        # In case the graphics backend changes, let's 
        # cache current backend and restore in the end function
        self.original_backend = mp.get_backend()
    
        # initialize loss file
        # create loss file if it does not exist
        if not os.path.exists(lossfile) or newlossfile:
            open(lossfile, 'w').write('iteration,train,val,valbest,lr\n')  
        
        self.reset()
        
    def __call__(self, t_loss, v_loss, lr=0, epoch=None, same_line=True):
        
        loss_decreased = v_loss < (1 - self.frac) * self.minavloss
        if loss_decreased:
            self.min_avloss = v_loss
        v_best_loss = self.minavloss
        
        # update loss file
        jj = self.ii-1 # the update occurs in  step()
        self.itno = self.offset + jj

        open(self.lossfile, 
             'a').write(f'{self.itno:10d},'
                        f'{t_loss:9.3e},{v_loss:9.3e},'
                        f'{v_best_loss:9.3e},{lr:9.3e}\n')

        if self.writer is not None:
            self.writer.add_scalar("Loss/train", t_loss, self.itno)
            self.writer.add_scalar("Loss/val",   v_loss, self.itno)
            self.writer.add_scalar("Loss/val_best", v_best_loss, self.itno)
            self.writer.add_scalar("LearningRate", lr, self.itno)
            self.writer.flush()
        
        # if specified save model parameters
        if type(self.model) != type(None):
            if loss_decreased:
                
                if os.path.exists(self.paramsfile):
                    cmd = f'mv {self.paramsfile} {self.paramsfile}.previous'
                    os.system(cmd)
                    
                self.model.save(self.paramsfile)

                if os.path.exists(self.checkfile):
                    cmd = f'mv {self.checkfile} {self.checkfile}.previous'
                    os.system(cmd)
                    
                open(self.checkfile, 'w').write(
                    'iteration,train,val,valbest,lr\n') 
                open(self.checkfile, 'a').write(
                    f'{self.itno:10d},'
                    f'{t_loss:9.3e},{v_loss:9.3e},'
                    f'{v_best_loss:9.3e},{lr:9.3e}\n')

        # update time left file
        if epoch != None:
            line = f'|{t_loss:9.3e}|{v_loss:9.3e}|{epoch:10d}|'
        else:
            line = f'|{t_loss:9.3e}|{v_loss:9.3e}|{self.itno:10d}|'
            
        timeleft_str = self.timeleft(jj, line)

        if same_line:
            print(f'\r{timeleft_str}', end='')
        else:
            print(timeleft_str)
            
        open(self.timeleftfile, 'w').write(f'{timeleft_str:s}\n')
        
    def step(self):
        save = self.ii % self.monitorstep == 0
        self.ii += 1
        return save

    def read_checkpoint(self):
        if sys.path.exists(self.checkfile):
            return pd.read_csv(self.checkfile)
        else:
            return None
        
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
        if self.writer is None:
            import subprocess

        self.reset()

        if self.writer is None:
            cmd = ["monlosses", self.lossfile]
            if self.ylabel != None:
                cmd.append(self.ylabel)
            print(' '.join(cmd))
            
            self.p = subprocess.Popen(cmd,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, 
                                      start_new_session=True  )

    def terminate(self):

        mp.use(self.original_backend, force=True)

        if self.writer is None:
            
            # kill standalone monitoring process
            try:
                print('\n\tTerminating standalone loss monitor...')        
                self.p.terminate()
                try:
                    self.p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.p.kill()
                    self.p.wait()
                print('\tDone!')
            except:
                print('\tNone stated in this session!')

        else:
            self.writer.flush()
            self.writer.close()