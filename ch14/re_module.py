# re_module.py

import re

p=re.compile("[a-z]+")
m=p.match("python")
print(m)

m=p.match("3 python")
print(m)                # None

print("------------------")
p=re.compile("[a-z]+")
m=p.match("python")

print(m)
print(type(m.group()))
print(type(m.span()))
print(type(m.start()))
print(type(m.end()))

if m:
    print(f"Match found: {m.group()}")
else:
    print('No match')

print("축약 전 형태-------")
p = re.compile("[a-z]+")
m = p. match("python")

print("축약 후 형태-------")
# m = p.match("정규표현식", "검색대상문자열")

m = re.match("[a-z]+", "python")
m = re.match("[a-z]+", "3python")
m = re.match("[a-z]+", "3py8thon")
print(m)