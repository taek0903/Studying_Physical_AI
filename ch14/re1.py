# re1.py

# 정규표현식(Regular Expression)
# 문자열 패턴을 정의해서 
# 검색,검사,치환,추출 등을 할 수 있게 해주는
# 문자열 처리 규칙

import re

# 정규표현식을 정의해서 컴파일 함수 인자로 전달하면
# 패턴 객체를 반환한다.
# 패턴객체: "검색 대상 문자열"에서 패턴이 발견을 도와주는 객체

# 패턴객체 = re.compile("정규표현식")
# 매치객체 = 패턴객체.match("검색대상문자열")
# print(매치객체)

# 메타문자 : 별도의 의미가 담겨있는 문자

# []
p = re.compile("[a]")
m = p.match("banana")
print(m)            # None

# p = re.compile("[abc]")   # 'b'
p = re.compile("[abc][abc]")
m = p.match("banana")
print(m)            # 'ba'

print('---------------------')
# p = re.compile("a")
# p = re.compile("b")
# p = re.compile("ab")      # None
p = re.compile("ba")        # 'ba'
m = p.match("banana")
print(m)      

print('---------------------')
p = re.compile("[a-c]")
m = p.match("before")       # 'b'                   
m = p.match("dude")         # None
print(m)

print('---------------------')
p = re.compile("[0-5]")
m = p.match("3banana")      # 3
print(m)

print('---------------------')
p = re.compile("[^0-5]")    # not "0,1,2,3,4,5"
print(p.match("3banana"))   # None
print(p.match("banana"))    # 'b'
print(p.match("^banana"))   # '^'

print('---------------------')
p = re.compile("[\^0-5]")   # '^' or "1, 2, 3, 4, 5" 일 경우 매치
print(p.match("3banana"))   # '3'
print(p.match("^banana"))   # '^'

print('---------------------')
p = re.compile("[a-z]")
print(p.match("apple"))     # 'a'
print(p.match("Apple"))     # None
print(p.match("banana"))    # 'b' 

print('---------------------')
# p = re.compile("[a-zA-Z0-9_]")
p = re.compile("[\w]")
# p = re.compile("[\W]")
print(p.match("apple"))     # 'a'
print(p.match("Apple"))     # None
print(p.match("banana"))    # 'b'
print(p.match("_melon"))    # '_'
print(p.match(" orange"))    # None

# 화이트스페이스
# \r = Carriage Return : 현재 줄 맨 앞으로 이동
# \f = form feed : 다음페이지로 넘김
# \v = vertical tab : 수직 텝

# p = re.compile("[a-zA-Z0-9_]")

print('---------------------')
p = re.compile("[\s]")
print(p.match("apple"))         # None
print(p.match("Apple"))         # None
print(p.match("banana"))        # None
print(p.match("_melon"))        # None
print(p.match(" orange"))       # ' '
print(p.match("\torange"))      # '\t'
print(p.match("\norange"))      # '\n'
print(p.match("\rorange"))      # '\r'

# .문자
# p = re.compile("a.b")
p = re.compile("a[.]b")
print(p.match("abb"))           # 'abb'         
print(p.match("a0b"))           # 'a0b
print(p.match("abc"))           # None
print(p.match("a\nb"))          # None 
print(p.match("a.b"))           # 'a.b'

print('---------------------')
# * 문자
p = re.compile("ca*t")
print(p.match("ct"))            # 'ct'
print(p.match("cat"))           # 'cat'
print(p.match("caaat"))         # 'caaat

print('---------------------')
# + 문자
p = re.compile("ca+t")          
print(p.match("ct"))            # None
print(p.match("cat"))           # 'cat'
print(p.match("caaat"))         # 'caaat'

print('---------------------')
# {} 문자
# {m}
# {m ,n}
p = re.compile("ca{2}t")          
print(p.match("ct"))            # None
print(p.match("caat"))          # 'caat'
print(p.match("caaat"))         # None

p = re.compile("ca{2,5}t")          
print(p.match("ct"))            # None
print(p.match("caat"))          # 'caat'
print(p.match("caaat"))         # 'caaat'

p = re.compile("ca{,5}t")          
print(p.match("ct"))            # 'ct'
print(p.match("caat"))          # 'caat'
print(p.match("caaat"))         # 'caaat'

p = re.compile("ca{2,}t")          
print(p.match("ct"))            # None
print(p.match("caat"))          # 'caat'
print(p.match("caaat"))         # 'caaat'

p = re.compile("ca{,}t")          
print(p.match("ct"))            # 'ct'
print(p.match("caaaaat"))       # 'caaaaat'

print('---------------------')
# ? = {0,1}
p = re.compile("ca?t")          
print(p.match("ct"))            # None
print(p.match("cat"))           # 'cat'
print(p.match("caaat"))         # None

print('---------------------')
# ^ 문자
p = re.compile("^hello")
print(p.match("taek,hello"))    # None
print(p.match("hello,taek"))    # 'hello'

print('---------------------')
# $ 문자
p = re.compile("hello$")
print(p.match("taek,hello"))    # None
print(p.match("taek, hello"))   # None
print(p.match("hello,taek"))    # None
print(p.match("kim,hello. taek,hello")) # None
print(p.match("hello"))         # 'hello'
# match 함수 : 검색을 맨 앞부터 일치하는지 검색

print('---------------------')
# serch 함수 : 문자열 전체에서 일치하는지 검색
#              일치하는 첫번째 문자열을 반환
p = re.compile("hello$")
print(p.search("taek,hello"))    # 'hello'
print(p.search("taek, hello"))   # 'hello'
print(p.search("hello,taek"))    # None
print(p.search("kim,hello. taek,hello")) # 'hello'
print(p.search("hello,kim. hello,taek")) # None

p = re.compile("hello")
print(p.search("taek,hello"))    # 'hello'
print(p.search("hello,taek"))    # 'hello'
print(p.search("kim,hello. taek,hello")) # 'hello'
print(p.search("hello,kim. hello,taek")) # 'hello'