# func3.py

# def funca():
#     print("funca 함수 호출")
# def funcb():
#     funca()
#     print("funcb 함수 호출")
# def funcc():
#     funcb()
#     print("funcc 함수 호출")
# funcc()

print("-----------------")
def fadd(pa, pb):
    pc= pa - pb
    return pc

na=10
nb=20
nc=fadd(na, nb)
print(na, "-", nb, "결과값은", nc, "이다.")

print("-----------------")
def draw_stars(num):
    print('*' * num)

draw_stars(3)
draw_stars(2)
draw_stars(1)

print("-----------------")
# def fadd(pa,pb):
#     pc = pa+pb
#     return pc
# def fsub(pa,pb):
#     pc = pa-pb
#     return pc
# def fmul(pa,pb):
#     pc = pa*pb
#     return pc
# def fdiv(pa,pb):
#     pc = pa/pb
#     return pc

# a= float(input('숫자를 입력하시오: '))
# b= float(input('숫자를 입력하시오: '))
# print(a, "+", b, "=", fadd(a,b))
# print(a, "-", b, "=", fsub(a,b))
# print(a, "x", b, "=", fmul(a,b))
# print(a, "/", b, "=", fdiv(a,b))
# print("-----------------")

sta = 'python example'
def string_length(stb):
    count = 0
    for char in stb:
        count += 1
    return count
lena =  string_length(sta)
print(lena)

print("-----------------")
def fdiv(a, b):
    if b == 0:
        return "0으로 나눌 수 없다."
    else:
        c= a/b
        return c

na=float(input("숫자를 입력하시오:"))
nb=float(input("숫자를 입력하시오:"))
print(fdiv(na, nb))