def count_down(n):
    if n==0:
        print("완료!")
        return
    print(n)
    count_down(n-1)

count_down(6)
print('------------------------')
x=10
def fadd(num):
    b=x+num
    print(f"변수x값은 {x}")
    print(f"변수b값은 {b}")
fadd(10)

print('------------------------')
x=10
def fadd(num):
    x=10
    x=x+num
    print(f"변수x값은 {x}")
fadd(10)

print('------------------------')
x=10
def fadd(num):
    global x
    x=x+num
    print(f"변수x값은 {x}")

# print('------------------------')
# def print_lower_price(price):
#     total = price-price*0.1
#     return print(total)
# a=int(input("현재가격을 입력하시오: "))
# print(print_lower_price(a))

def func1(num):
    return num+4
a=func1(10)
b=func1(a)
c=func1(b)
print(c)