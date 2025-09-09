#!/usr/bin/env python
#------------------------------------------------------------------------------
# Real time monitoring of loss curves during training
# Harrison B. Prosper
# July 2021
# Aug  2025 HBP use mlinphysics
#------------------------------------------------------------------------------
import os, sys
import mlinphysics.utils.monitor as mon
#------------------------------------------------------------------------------
def main():
    # get name of loss file
    argv = sys.argv[1:]
    argc = len(argv)
    if argc < 1:
        sys.exit('''
        Usage:
           ./monitor_losses.py loss-file
    ''')
        
    loss_file = argv[0]
    print()
    print('loss file:', loss_file)

    monitor = mon.Monitor(loss_file)
    
    monitor()
#------------------------------------------------------------------------------
try:
    main()
except KeyboardInterrupt:
    print('\nciao!\n')

