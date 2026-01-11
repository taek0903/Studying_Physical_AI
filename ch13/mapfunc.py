# mapfunc.py

def squre(x):
    return x**2
print(squre(2))

numbers = [1,2,3,4,5]

# 이터레이터 = map(함수, 이터러블)
#  => 목록자료 = map(함수, 시퀀스자료)
squared_numbers = map(squre, numbers)
print(type(squared_numbers))
print(squared_numbers)
print(list(squared_numbers))

print('------------------')
square = lambda x : x**2
squared_numbers = map(square, numbers)
print(list(squared_numbers))

squared_numbers = map(lambda x : x**2, numbers)
print(list(squared_numbers))

print('------------------')
numList=['010',1234,'5678']
joined_string="-".join(map(str,numList))
print(joined_string)