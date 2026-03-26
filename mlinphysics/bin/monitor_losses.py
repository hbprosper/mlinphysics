#!/usr/bin/env python
#------------------------------------------------------------------------
# Real time monitoring of loss curves during training
# Harrison B. Prosper
# July 2021
# Aug  2025 HBP use mlinphysics
#------------------------------------------------------------------------
import os, sys
import mlinphysics.utils.monitor as mon
#------------------------------------------------------------------------
def main():
    # get name of loss file
    argv = sys.argv[1:]
    argc = len(argv)
    if argc < 1:
        sys.exit('''
        Usage:
           monlosses loss-file [ylabel] [ylog=1]
    ''')
        
    lossfile = argv[0]
    if argc > 1:
        ylabel = argv[1]
    else:
        ylabel='$R(\\omega)$'

    if argc > 2:
        ylog = int(argv[2])
    else:
        ylog = 1 
        
    monitor = mon.LossMonitor(lossfile, ylabel, ylog)
    monitor.show()


