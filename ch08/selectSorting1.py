# selectSorting1.py

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