import os
import subprocess

import inotify.adapters

OUT_DEVICE_NUMBER = 5

def run_watcher():
    i = inotify.adapters.Inotify()
    mask = inotify.constants.IN_OPEN | inotify.constants.IN_CLOSE
    i.add_watch('/dev/video' + str(OUT_DEVICE_NUMBER), mask=mask)

    p = []
    for event in i.event_gen(yield_nones=False):
        (_, types_names, _, _) = event
        
        if 'IN_OPEN' in types_names:
            proc = subprocess.Popen(["python3","main.py"])
            p.append(proc)
            # print("open", str(p))
        
        if 'IN_CLOSE_NOWRITE' in types_names or 'IN_CLOSE_WRITE' in types_names:
            # print("killing", str(p))
            for proc in p:
                proc.kill()
            p = []


if __name__ == '__main__':
    run_watcher()
