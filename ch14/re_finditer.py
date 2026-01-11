# re_finditer.py

import re

# 패턴객체 = re.compile("정규표현식")
# 매치객체 = 패턴객체.finditer("검색대사문자열")

p=re.compile("[a-z]+")

# print(p.finditer("3pyt8hon"))  
# print(p.finditer('life is too short'))  

print(list(p.finditer("3pyt8hon")))  # ['pyt', 'hon']
# [<re.Match object; span=(1, 4), match='pyt'>, 
#  <re.Match object; span=(5, 8), match='hon'>]

print(list(p.finditer('life is too short')))  # ['life', 'is', 'too', 'short']
# [<re.Match object; span=(0, 4), match='life'>, 
# <re.Match object; span=(5, 7), match='is'>, 
# <re.Match object; span=(8, 11), match='too'>, 
# <re.Match object; span=(12, 17), match='short'>]

matchObjs = p.finditer('life is too short')
for matchObj in matchObjs:
    print(matchObj.group())
    # print(matchObj.span())
    # print(matchObj.start())
    # print(matchObj.end())


