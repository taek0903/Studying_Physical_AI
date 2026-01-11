# p2.py
from collections import deque

queue=[1,2,3,4,5]
def rotate_queue(queue,k):
    dq = deque(queue)
    # for i in range(k):
    #     dq.append(dq.popleft())
    dq.rotate(-k)
    return list(dq)

print(rotate_queue(queue,2))
    