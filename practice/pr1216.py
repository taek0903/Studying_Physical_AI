# 1. 두 변수 값 바꾸기
a = 3
b = 10
a, b= b, a
print(a,b)
print('-------------')

# 2. 최댓값 반환 함수 만들기
def get_max(a, b):
    if a > b:
        print(a)
    elif a < b:
        print(b)
get_max(7,2)
get_max(2,7)
print('-------------')

# 3. 리스트에서 짝수만 골라서 새 리스트 만들기
def check_odd_even(list):
    even=[]
    for i in list:
        if i % 2 == 0:
            even.append(i)
    print(even)
num=range(0,11)
check_odd_even(num)
print('-------------')

# 4. 문자열 뒤집기 함수 만들기
# def reverse_str(s):
#     return s[::-1]

# s= input().strip()
# print(reverse_str(s))

# 5. 간단한 계산기 함수
def clac(a,b,op):
    if op == "+":
        return print(a+b)
    elif op == "-":
        return print(a-b)
    elif op == "*":
        return print(a*b)

clac(2,3,"*")     
print('-------------')

# 6. 리스트에서 가장 큰수와 가장 작은 수를 반환하는 함수
def get_min_max(lst):
    a=(max(lst),min(lst))
    return print(a)
get_min_max([3, 7, 1, 9, 4])

# 7. 문자열에서 모음 개수 세기
def count_vowels(s):
    vowels = 'aeiouAEIOU'
    count = 0
    for ch in s:
        if ch in vowels:
            count += 1
    return count
s= input().strip()
print(count_vowels(s))
print('-------------')

# 8. 중복제거
def remove(lst):
    result = []
    for n in lst:
        if n not in result:
            result.append(n)
    return result

nums = list(map(int, input().split()))
print(remove(nums))
print('-------------')

# 9. 회문판별
def is_palindrome(s):
    if s == s[::-1]:
        return "YES"
    return "NO"
s = input().split()
print(is_palindrome(s))
print('-------------')

# 10. 짝수 홀수 나누기
def split_even_odd(lst):
    odd=[]
    even=[]
    for i in lst:
        if i % 2 == 0:
            even.append(i)
        else:
            odd.append(i)
    return print(odd), print(even)

a=range(0,21)
split_even_odd(a)