# comprehension.py

# 컴프리헨션
# 시퀀스 자료형을 생성한다. => 잘못된 표현
# 시퀀스(순서)를 생성한다. => 정확한표현
# 시퀀스(순서) == 순서가 있고 반복가능한 데이터
# 딕셔너리에도 순서가 있다. (3.7+)

gen = (i*i for i in range(1,10))

print(next(gen))
for item in gen:
    print(item)
# print(next(gen))        #StopIteration

print('----------------------')
# 리스트 컴프리핸션
# [표현식 for 요소 in 이터러블 if 조건]

numbers = [1,2,3,4]
squared = [i*i for i in numbers]
print(squared)
print(type(squared))

# 조건문이 참인 경우, 요서를 포함하여 표현식 수행
squared = [i for i in numbers if i%2==0]
print(squared)
squared = [i*i for i in numbers if i%2==0]
print(squared)
squared = [i*i+1 for i in numbers if i%2==0]
print(squared)

print('----------------------')
# 딕셔너리 컴프리핸션
# {key:value, key:value, key:value}
# [key표현식:value표현식 for 요소 in 이터러블 if 조건]

squared_dict = {x:x**2 for x in range(5)}
print(type(squared_dict))
print(squared_dict)

squared_dict = {x:x**2 for x in range(5) if x%2 ==0}
print(squared_dict)

subjects =['math', 'english', 'history']
scores = {subject:0 for subject in subjects}
print(scores)

print('----------------------')
# 제너레이터 컴프리핸션
# 제너레이트 생성 방법 2
# (표현식 for 요소 in 이터러블 [if 조건])

gen = (i*i for i in range(1,10))

print(next(gen))
for item in gen:
    print(item)

print('----------------------')
# 시퀀스(순서가 있다)
dict1 = {'a':1, 'b':2, 'c':3}
del dict1['b']
dict1["b"]=2
print(dict1)

# 시퀀스 자료형
from collections.abc import Sequence
dict2 = {'a':1, 'b':2, 'c':3}
print(isinstance(dict2,Sequence))