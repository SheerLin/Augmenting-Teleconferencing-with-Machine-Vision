import argparse
import os
import sys
import subprocess

from main import parse_args
import inotify.adapters

CAP_DEVICE_NUMBER = 2 # output device

def run_watcher():
    i = inotify.adapters.Inotify()

    # Add watcher
    mask = inotify.constants.IN_OPEN | inotify.constants.IN_CLOSE
    i.add_watch('/dev/video' + str(CAP_DEVICE_NUMBER), mask=mask)

    p = []
    for event in i.event_gen(yield_nones=False):
        (_, types_names, _, _) = event
        
        if 'IN_OPEN' in types_names:
            # Create
            proc = subprocess.Popen(['python3','main.py'] + sys.argv[1:])
            p.append(proc)
        
        if 'IN_CLOSE_NOWRITE' in types_names:
            # Kill one
            if len(p) > 0:
                proc = p.pop()
                proc.kill()

        if 'IN_CLOSE_WRITE' in types_names:
            # Kill all
            for proc in p:
                proc.kill()
            p = []

    # Remove watcher
    i.remove_watch('/dev/video' + str(CAP_DEVICE_NUMBER))

if __name__ == '__main__':
    args = parse_args()
    CAP_DEVICE_NUMBER = args.out
    run_watcher()
