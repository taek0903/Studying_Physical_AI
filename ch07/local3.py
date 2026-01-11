# local3.py

b=0
print(f"b의 값 {b}")
b=1
print(f"b의 값 {b}")

def scope_test():
    # global a
    a = 0
    print(f"a 값: {a}")
    a = a+3  # 지역변수
    print(f"함수 안의 a 값: {a}")
    
a=0 # 전역변수
print(f"함수 밖의 a 값: {a}")
scope_test()
print(f"함수 호출 후 a 값: {a}")