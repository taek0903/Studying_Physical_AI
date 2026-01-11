# Human.py

class Human:
    # 속성 => 맴버변수
    eyes = 2
    nose = 1
    mouth = 1
    def __init__(self, name, age):
        self.name = name
        self.age = age
    # 기능/동작 => 메서드
    def introduce(self):
        print(f"이름: {self.name} 나이: {self.age}")
    def eat(self):
        print("먹다")
    def sleep(self):
        print("자다")
    def talk(self):
        print("말하다")
    def breath(self):
        print('숨쉬다')

class Student(Human):
    def __init__(self, name, age, studentNum):
        # self.name = name
        # self.age = age
        super().__init__(name, age)
        self.studentNum=studentNum
    def introduce(self,):
        super().introduce()
        print(f"학번: {self.studentNum}")
    def study(self):
        print("공부하다")
    def exam(self):
        print("시험보다")

class Streamer(Human):
    def streaming(self):
        print("방송하다.")
    def costreaming(self):
        print("합방하다.")

lee = Human('이수근',49)
lee.introduce()
lee.eat()
lee.sleep()
print(lee.eyes)
print('----------------------')
kim = Student('김수로',56,20251219)
kim.introduce()
kim.talk()
kim.study()
kim.exam()
print(kim.studentNum)
print(kim.nose)
print('----------------------')
big=Streamer('최필립',32)
big.introduce()
big.streaming()

