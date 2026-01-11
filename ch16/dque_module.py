# dque_module.py
from collections import deque

dq=deque()          # 덱 생성

dq.append(1)        # dq에 뒤로 데이터 넣기
dq.append(3)
print(dq)

dq.appendleft(2)    # dq에 앞으로 데이터 넣기
print(dq)

dq.pop()            # 마지막 데이터 꺼내기
print(dq)

dq.popleft          # 처음 데이터 꺼내기
print(dq)

print(type(dq))