# re_search.py

import re

# 패턴객체 = re.compile("정규표현식")
# 매치객체 = 패턴객체.search("검색대사문자열")

p=re.compile("[a-z]+")
m=p.search("python")
print(m)

m=p.search("3 python")  # 'python'
m=p.search("3pyt8hon")  # 'pyt'
print(m)                # 'python'

print(m.group())
print(m.span())
print(m.start())
print(m.end())


print("------------------")

text = "The Python Software Foundation"
p=re.compile('Python')
print(p.search(text))        # None

p=re.compile('[a-z]*')
print(p.search('3pyt8hon'))  # None
# 아무것도 없어도 조건 만족(0글자 매칭 가능)
