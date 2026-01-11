# list1.py

candy0 = '딸기맛'
print(candy0)
candy1 = '레몬맛'
print(candy1)
candy2 = '수박맛'
print(candy2)
candy3 = '박하맛'
print(candy3)
candy4 = '우유맛'
print(candy4)
print(type(candy0))

candies = ['딸기맛', '레몬맛', '수박맛', '박하맛', '우유맛']
print(candies)
print(type(candies))

lista = ['list', 1, 0.7, True]
print(lista)
print(type(lista))

print("------------------")
ca=[10,11,21]
print(ca)
print(type(ca))

print(ca[0], end=", ")
print(ca[1], end=", ")
print(ca[2])

print(type(ca[0]))

print("------------------")
lista = [1, "python", 3.9, '프로그래밍', [10, 11, 21]]
print(lista)
print(lista[0])
print(lista[1])
print(lista[2])
print(lista[3])
print(lista[4])
print(type(lista[4]))

print(lista[4][2])
print(type(lista[4][2]))

lista[0]=5
print(lista)

print("------------------")
a=[1, 2, 3, 4, 5]
a[2]=30
print(a)
a[3]=40
print(a)
a[1]='hi'
print(a)
a[4]=False
print(a)
a[0]=3.14
print(a)

# print(a[5])
# a[5] = 'thanks' => 범위를 벗어나면 무조건 오류 => 추가되는 명령어가 아님
