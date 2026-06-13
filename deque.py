import torch
from collections import deque
len=4
box = deque(maxlen=len)
print(box)
box.append(1)
print(box)
box.append(2)
print(box)
box.append(3)
print(box)
box.append(4)
print(box)
box.append(5)
print(box)