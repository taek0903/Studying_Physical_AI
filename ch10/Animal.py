# Animal.py

class Animal:
    def eat(self):
        print("먹다")
    def move(self):
        print("이동하다")
    def sound(self):
        print("소리내다")

class Cat(Animal):
    def eat(self):
        super().eat()
        print("쥐를 잡아먹다.")
    def sound(self):
        print("야옹~ ", end="")
        super().sound()

animal = Animal()
animal.eat()
animal.sound()
print('------------------')
cat1 =  Cat()
cat1.eat()
cat1.move()
cat1.sound()
print('------------------')

### 다양한 문자열 출력 방법
# 표준 출력 함수 : print(인수) 활용
# 1. 기능 : 인수 출력
# 2. 인수 : 객체(다양한 자료형 모두)
# 3. 리턴값 없음

# 기본형
# print("이름", name, "나이", age)

# print 함수 사용법

# 1. % 연산자 사용 방식
# 서식문자 활용 => %s:문자열, %d:10진수, %f :실수형, %x :16진수
# print("이름:%s 나이%d"%(name,  age))

# 2. format() 매서드 사용 방식
# print("이름:{} 나이:{}".format(name,  age))

# 3. f-string 사용 방식
# print(f"이름:{name} 나이:{age}")