# hw16

print('6----------------')
class ListStack:
    def __init__(self):
        self.stack=[]
    
    def push(self, data):
        self.stack.append(data)
    
    def pop(self):
        if self.is_empty():
            return -1
        return self.stack.pop()
    
    def is_empty(self):
        if len(self.stack) == 0:
            return True
        return False
    
    def top(self):
        if self.is_empty():
            return -1
        return self.stack[-1]

s1 = ListStack()
print(s1.top())
print(s1.pop())
s1.push(1)
print(s1.top())

print('7----------------')
class PostfixNotation:
    def __init__(self,data):
        self.tokens=data.split()
        self.stack=[]
    
    def evaluate(self):
        for tok in self.tokens:
            if tok.lstrip("-").isdigit():
                self.stack.append(int(tok))

            elif tok in "+-*/":
                if len(self.stack) < 2:
                    return -1
                b = self.stack.pop()
                a = self.stack.pop()
                if tok == '+':
                    self.stack.append(a+b)
                elif tok == '-':
                    self.stack.append(a-b)
                elif tok == '*':
                    self.stack.append(a*b)
                elif tok == '/':
                    if b == 0 :
                        return -1
                    self.stack.append(int(a/b))
            else:
                return -1
        if len(self.stack)!=1:
            return -1
        return self.stack[0]

p = PostfixNotation("3 4 +")
print(p.evaluate())  
p = PostfixNotation("3 4 + 2 *")
print(p.evaluate()) 
p = PostfixNotation("5 1 2 + 4 * + 3 -")
print(p.evaluate()) 

print('8----------------')
class Queue:
    def __init__(self):
        self.queue=[]
    def enqueue(self, data):
        self.queue.append(data)
    def dequeue(self):
        if self.is_empty():
            return -1
        return self.queue.pop(0)
    def front(self):
        return self.queue[0]
    def is_empty(self):
        if len(self.queue) == 0:
            return True
        return False
q= Queue()
print(q.dequeue())
q.enqueue(10)
q.enqueue(20)
print(q.dequeue())
q.enqueue(30)
print(q.front())

print('9----------------')
class FixQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.q=[None] * capacity
        self.front_idx=0
        self.rear_idx=0
        self.count=0
    def is_empty(self):
        return self.count==0
    def is_full(self):
        return self.count==self.capacity
    def enqueue(self, x):
        if self.is_full():
            return False
        self.q[self.rear_idx]=x
        self.rear_idx=(self.rear_idx+1)%self.capacity
        self.count +=1
        return True
    def dequeue(self):
        if self.is_empty():
            return -1
        value = self.q[self.front_idx]
        self.q[self.front_idx]=None
        self.front_idx=(self.front_idx+1)%self.capacity
        self.count-=1
        return value
    def front(self):
        if self.is_empty():
            return -1
        return self.q[self.front_idx]
cq = FixQueue(3)   
print(cq.dequeue())     
print(cq.front())       

print(cq.enqueue(10))   
print(cq.enqueue(20))   
print(cq.enqueue(30))   
print(cq.enqueue(40))   
print(cq.q)
print(cq.front())       
print(cq.dequeue())     
print(cq.dequeue())     

print(cq.enqueue(40))   
print(cq.front())  

print('10---------------')
class ListDeque:
    def __init__(self):
        self.deque=[]
    def push_front(self,x):
        self.deque.insert(0,x)
    def push_back(self,x):
        self.deque.append(x)
    def pop_front(self):
        if self.is_empty():
            return -1
        return self.deque.pop(0)
    def pop_back(self):
        if self.is_empty():
            return -1
        return self.deque.pop()
    def is_empty(self):
        return len(self.deque) == 0
d = ListDeque()
print(d.pop_front())  

d.push_back(10)
d.push_front(20)
d.push_back(30)

print(d.pop_front()) 
print(d.pop_back())   
print(d.pop_back())   
print(d.pop_back())   
