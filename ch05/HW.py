# HW.py

# 1번
fruit = ["사과", "귤", "수박"]
for x in fruit:
    print(x)
print("--------------------")    

# 2번
i = [10, 20, 30]
for x in i:
    print(x)
print("--------------------")

# 3번
price=[100, 200, 300]
for x in price:
    print(x+10)
print("--------------------")

# 4번
animal = ['dog', 'cat', 'parrot']
for y in animal:
    print(y, len(y))
print("--------------------")    

# 5번
char = ["가","나","다","라"]
for y in char:
    if y == "가":
        continue
    print(y)
print("--------------------")

# 6번
num1 = [3, -20, -3, 44]
for y in num1:
    if y < 0:
        print(y)
print("--------------------")

# 7번
for y in range(2002,2051):
    if (y-2002) % 4 == 0:
        print(y, end=" ")
print(end="\n")
print("--------------------")

# 9번
num2 = 1
total = 0
while num2 <=100:
   total += num2
   num2 += 1
print(total)
print("--------------------")

# 10번
for z in range(1,31):
    if z % 2 != 0:
        print(z,": 홀수",end=" ")
    else:
        print(z,": 짝수",end=" ")
print(end="\n")
print("--------------------")

# 11번
list1 = []
list2 = []
for z in range(1,31):
    if z % 2 != 0:
        list1.append(z)
    else:
        list2.append(z)
print("홀수 리스트: ",list1)
print("짝수 리스트: ",list2)
print("--------------------")