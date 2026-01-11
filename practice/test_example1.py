print('1------------------')
for i in range(1,6):
    print(i)

print('2------------------')
for i in range(0,10):
    if i % 2 == 0:
        print(i)

print('3------------------')
a=[10,20,30]
x=a.pop()
print(x)

print('4------------------')
d = {'name': 'Rokey', 'age':3}
print(d['name'])

print('5------------------')
s='   python'
t=s.strip()
print(t)

print('6------------------')
s='I love python'
words=s.split()
print(words)

print('7------------------')
try:
    x = int('가')
    print(x)
except ValueError:
    print(0)

print('8------------------')
def add(a,b):
    return a+b

print('9------------------')
def squares(n):
    for i in range(1,n+1):
        yield i*i
sq=squares(5)
for i in sq:
    print(i)

print('10-----------------')
stack=[1,2,3]
v=stack.pop()
print(v)

print('-----------절취선-------------')

print('1------------------')
nums = [1,2,3,4,5]
result = [x for x in nums if x%2 !=0]
print(result)

print('2------------------')
d={'a':1,'b':2,'c':3}
filtered = {k:v for k,v in d.items() if v >=2}
print(filtered)

print('3------------------')
s = '     a, b, c'
items = [x.strip() for x in s.split(',')]
print(items)

print('4------------------')
try:
    int(10)
except ValueError:
    print('ERR')
else:
    print('OK')

print('5------------------')
from collections import deque
q = deque([1,2,3])
v = q.popleft()
print(v)

print('6------------------')
lst=['a','b','c']
for i,v in enumerate(lst):
    print(i,v)

print('7------------------')
def outer():
    x = 10
    def inner():
        nonlocal x
        x += 1
        return x
    return inner()

f=outer()
print(f)

print('8------------------')
def add_item(x, items=None):
    if items is None:
        items = []
    items.append(x)
    return items
print(add_item(10))

print('9------------------')
freq={}
for ch in 'banana':
    freq[ch]=freq.get(ch,0)+1
print(freq)

print('10-----------------')
def squares(n):
    for i in range(1,n+1):
        yield i*i

a=squares(3)
for i in a:
    print(i)
