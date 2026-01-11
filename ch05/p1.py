#p1.py

# # 1번
# for num in [3,1,2]:
#     print(num)
# print("---------------")

# # 2번
# for num in range(2):
#     print(num)
# print("---------------")

# # 3번
# clovers = ["클로버1","클로버2", "클로버3"]
# for clover in clovers:
#     print(clover)
# print("---------------")

# # 4번
# clovers = ["클로버1", "클로버2", "클로버3"]
# for j in range(len(clovers)):
#     print(clovers[j])
# print("---------------")

# # 5번
# count = 0
# while count < 3:
#     print(count)
#     count = count+1
# print("---------------")

# # 6번
# count = 1
# while count < 4:
#     count += 1
#     print(count)
# print("---------------")

# # 7번
# count = 0
# while count <= 5: # 0 1 2 3 4 5
#     if count % 2 !=0:
#         print(count)
#     count += 1
# print("---------------")
# # 출력 결과
# # 1 3 5

# 8번
# price = 0
# while price != -1 :
#     price = int(input('가격을 입력하세요 (종료:-1)'))
#     if price > 10000:
#         print('너무 비싸요.')
#     elif price > 5000:
#         print('괜찮은 가격이네요.')
#     elif price > 0:
#         print('정말 싸요.')
# print("---------------")

# # 9번
# num = 0
# while num <= 100 :
#     if num % 3 == 0:
#         print(num, end=", ")
#     num += 1
# print(end="\n")
# print("---------------")

# # 9-2
# for num in range(101):
#     if num % 3 == 0:
#         print(num, end=" ")
# print(end="\n")
# print("---------------")

# # 10번
# num = 1
# while num <= 10:
#     if num % 3 != 0:
#         print(num, end=" ")
#     num += 1
# print(end="\n")
# print("---------------")

# # 10-2
# for num in range(1,11):
#     if num % 3 != 0:
#         print(num, end=" ")
# print(end="\n")
# print("---------------")

# # 11번
# for num in range(1,11):
#     if num % 3 == 0:
#         continue
#     print(num, end=", ")
# print(end="\n")
# print("---------------")

# 11-2
# num = 0
# while num <= 9:
#     num = num +1
#     if num % 3 == 0:
#         continue
#     print(num, end=" ")
# print(end="\n")
# print("---------------")

# 12
# num = int(input("총합을 구하려는 수를 입력하시오."))
# total = 0
# for i in range(0,num+1):
#     total +=i
# print("1부터", num, "까지의 총합은", total, "입니다.")
# print("---------------")

# num = int(input("총합을 구하려는 수를 입력하세요."))
# total = 0
# j = 1
# while j <= int(num):
#     total += j
#     j += 1
# print("1부터", num, "까지의 총합은", total, "입니다.")
# print("---------------")

# # 33pg
# for o in range(1,10):
#     print("1 X",o,"=",o)

# # 35pg
# for p in range(1,6):
#     for q in range(1,10):
#         print(p, "X", q, "=", p*q)

# 37pg
for x in range(1,6):
    for y in range(1,x+1):
        print(y,end="")
    print(end="\n")

for x in range(1,6):
    for y in range(x):
        print(chr(ord('A')+y),end="")
    print(end="\n")