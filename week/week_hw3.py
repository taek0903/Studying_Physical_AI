print('12-2----------------')
with open("data.text", 'w', encoding='utf-8') as file:
    for i in range(1,11):
        file.write(f"{i}번쨰 줄입니다.\n")
print('파일이 성공적으로 작성되었습니다.')
with open('data.text', 'r', encoding='utf-8') as file:
    contents = file.read()

print('파일내용: ')
print(contents)

with open("data.text", 'a', encoding='utf-8') as file:
    data = "11번째 줄입니다."
    file.write(data)

print(contents)

print('13-1----------------')
while True:
    try:
        x=int(input("숫자를 입력해주세요: "))
        result = x ** 2
        print(result)
        break
    except ValueError :
        print("올바른 숫자를 입력하세요!")

print('13-2----------------')
print(list(map(lambda x : x **2 , [10,20,30,40,50])))

print('14-1----------------')
import re
data = """python one
life is too short
python two
you need python
python three"""
p = re.compile("[^.\n]*python[^.\n]*",re.MULTILINE)
print(p.findall(data))

print('14-2----------------')
p = re.compile('[a-zA-Z0-9_+*%-]+@[a-zA-Z0-9_+*%-]+\.[a-zA-Z0-9_+*%-]{2,}')
data=input("이메일을 입력하세요.")
print(p.findall(data))