# re.findall.py

import re

# 패턴객체 = re.compile("정규표현식")
# 매치객체 = 패턴객체.findall("검색대사문자열")

p=re.compile("[a-z]+")

print(p.findall("3 python"))  # ['python']
print(p.findall("3pyt8hon"))  # ['pyt', 'hon']
print(p.findall('life is too short'))  # ['life', 'is', 'too', 'short']

print("------------------")

text = "The Python Software Foundation welcomes Python Python" 
p=re.compile('Python')
print(p.findall(text))        # # ['Python', 'Python', 'Python']

p=re.compile('[a-z]*')
print(p.findall('3pyt8hon'))  # ['', 'pyt', '', 'hon', '']
# 아무것도 없어도 조건 만족(0글자 매칭 가능)


