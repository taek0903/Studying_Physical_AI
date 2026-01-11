# selectSorting2.py

ca=[21,10,11,15,13]
mina=ca[0]
minix=0
for sb in range(1,5,1):
    if mina > ca[sb]:
        mina = ca[sb]
        minix = sb
print(ca)

temp= ca[0]
ca[0]= ca[minix]
ca[minix]=temp
print(ca)
# 1차 목표 : [10,21,11,15,13]

print('-------------------')
mina= ca[1]
minix = 1

for sb in range(2,5,1):
    if mina > ca[sb]:
        mina = ca[sb]
        minix = sb
print(ca)

temp = ca[1]
ca[1]=ca[minix]
ca[minix]=temp
print(ca)
# 2차 목표 : [10, 11, 21, 15, 13]

print('-------------------')
mina = ca[2]
minix = 2

for sb in range(3,5,1):
    if mina > ca[sb]:
        mina = ca[sb]
        minix = sb
print(ca)

temp=ca[2]
ca[2]=ca[minix]
ca[minix]=temp
print(ca)
# 3차 목표 : [10, 11, 13, 21 ,15]