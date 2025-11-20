#!/usr/bin/env python
#------------------------------------------------------------------------------
# Real time monitoring of loss curves during training
# Harrison B. Prosper
# July 2021
# Aug  2025 HBP use mlinphysics
#------------------------------------------------------------------------------
import os, sys
import matplotlib.pyplot as plt
import mlinphysics.utils.monitor as mon
#------------------------------------------------------------------------------
anim = None
def main():
    global anim
    # get name of loss file
    argv = sys.argv[1:]
    argc = len(argv)
    if argc < 1:
        sys.exit('''
        Usage:
           monlosses loss-file
    ''')
        
    loss_file = argv[0]
    monitor = mon.Monitor(loss_file, init_fig=True)
    anim = monitor()    
    plt.show()


