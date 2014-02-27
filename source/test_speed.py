# This file is simply here to make sure that everything is running just as
# fast under the virtualbox as under the host OS. There should be no
# performance degradation. This takes me (Ben) approximately 1.2sec outside my
# virtual machine and appx 1.15sec inside the virtual machine. WEIRD IT IS
# FASTER! WHY?

import time

def time_fnc():
    a = range(1, 1000000)
    for i in range(1, 200):
        b = sum(a)

t0 = time.time()
time_fnc()
t1 = time.time()
print(t1 - t0)
