# hw15.py

print('6------------------')
numbers=[1,2,3,4,5]
inumbers=iter(numbers)
for i in inumbers:
    print(i)

print('7------------------')
fruits = ['apple', 'banana', 'cherry']
class MyIterator():
    def __init__(self, data):
        self.data = data
        self.position=0
    def __iter__(self):
        return self
    def __next__(self):
        if self.position >= len(self.data):
            raise StopIteration
        result = self.data[self.position]
        self.position += 1
        return result

ifruits = MyIterator(fruits)
for i in ifruits:
    print(i)

print('8------------------')
squred=(x**2 for x in range(0,10))
for i in squred:
    print(i)

print('9------------------')
even_squred=(x**2 for x in range(0,10) if x % 2 ==0)
for i in even_squred:
    print(i)

print('10-----------------')
class MyRange():
    def __init__(self,start,stop,step=1):
        if step == 0:
            raise ValueError
        self.start=start
        self.stop=stop
        self.step=step
        self.current= start
    
    def __iter__(self):
        return self
   
    def __next__(self):
        if self.step > 0:
            if self.current >= self.stop:
                raise StopIteration
        else:
            if self.current <= self.stop:
                raise StopIteration
        value = self.current
        self.current += self.step        
        return value

a = MyRange(0,10,1)
for i in a:
    print(i)