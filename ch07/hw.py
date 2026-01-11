# hw.py

# 3번
def calculate_area(length, width=10):
    return length*width
print(calculate_area(5))
print(calculate_area(5,20))
# calculate_area()함수 매개변수 width의 기본값이 10으로 
# 저장되어 있으므로 print(calculate_area(5))을 호출하면 5*10 50의 값이 나온다.
# print(calculate_area(5,10))을 호출하면 width 기본값 10 대신 20이 들어가
# 5*20 =100의 값이 나온다.
print("------------------")

# 4번
# add_numbers(a,b)의 함수의 매개변수는 2개이지만
# print(add_numbers(10))로 호출할 때는 한개의 인자만 들어가있기 때문입니다.
def add_numbers(a,b):
    return a + b
print(add_numbers(10,10))
print("------------------")

# 5번
def inner_function(x,y):
    return x+y
def outer_function(x,y):
    return inner_function(x,y)
add_10=outer_function(10,5)
print(add_10)
# 1. outer_function에 인수 10과 5가 할당이 된다.
# 2. inner_function으로 10과 5가 할당이 된후 10+5의 값이 반환된다.
# 3. inner_function에서 반환된 15의 값이 outer_funcion에 반환된다.
# 4. outer_function에서 반환된 15의 값의 add_10에 할당된다.
print("------------------")

# 6번
def add_numbers(a,b):
    result = a + b
    return result
result=add_numbers(10,5)
print(result)
# add_numbers(a,b) 함수의 반환값이 존재하지 않았습니다.
# result 변수는 함수 내에서 작용되는 지역변수이기 때문에
# 함수가 종료되면 사라지기 때문입니다.
print("------------------")

# 7번
def message():
    print("A")
    print("B")
message()
print("C")
message()
print("------------------")

# 8번
print("A")
def message():
    print("B")
print("C")
message()
print("------------------")

# 9번
def check_odd_even(a):
    if a % 2 != 0:
        return "Odd"
    else:
        return "Even"
print(check_odd_even(4))
print(check_odd_even(7))
print("------------------")

# 10번
num_list=[10,20,30,40,50]
def calculate_average(num_list):
    average = 0
    total = 0
    for i in num_list:
        total += i
    average = total / len(num_list)
    print(f"평균: {average}")
calculate_average(num_list)