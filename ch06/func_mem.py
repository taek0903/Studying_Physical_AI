# func_mem.py

# na = 10
# nb = 11

# pa = na
# pb = nb

# nc = pa + pb
# print("pa: ", pa)
# print("pb: ", pb)
# print("nc: ", nc)
# print("na: ", na)
# print("nb: ", nb)
# print("-------------------")
# # na = 30
# # nb = 31
# pa = 12
# pb = 13
# print("pa: ", pa)
# print("pb: ", pb)
# print("na: ", na)
# print("nb: ", nb)

print("-------------------")
def funca(pa, pb):
    nc = pa + pb
    return nc
na = 10
nb = 11

nd = funca(na, nb)
print(na, "+", nb, "=", nd)
# 함수 내 처리한 결과는 전체 영역에서 사용하기 위해서 반환(return)해야한다.
# 반환값은 다양한 자료형 사용 가능
# 반환값은 여려개의 데이터 사용 가능
# return 단독 사용 : 키워드 이후 정의된 함수 코드 블록 실행되지 않음

print("-------------------")
def fplusminus(arg):
    if arg > 0:
        return "plus"
    elif arg < 0:
        return "minus"
stra = fplusminus(-1)
print(stra)