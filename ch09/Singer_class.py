# Singer_class.py
# 클래스(분류:가수) 정의
class Singer:
    def __init__(self, name):
        self.name = name
    # name = "아이유"      # (클래스)맴버변수 -> 공용정보 우선처리
    def sing(self):      # 맴버함수(매서드)
        # self.name = "아이유2"  # (인스턴스)맴버변수
        print("이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요~")

# 객체 생성(생성자 함수 활용)
iu = Singer("아이유")
print(iu.name)  # 객체 속성 접근
# print(Singer.name) # 클래스 맴버 변수 접근 -> 공용정보 우선처리
iu.sing()       # 객체 기능/동작 접근
print('-----------------------')
print(iu.name)  # 객체 기능/동작 접근 -> 아이유2가 나온다.
# print(Singer.name)

bts = Singer("bts")
print(bts.name)
# bts.sing()

# 클래스 맴버 변수: 클래스(분류)의 공용 정보 처리 
# 인스턴스 맴버 변수: 각 객체(실체)의 개별 정보 처리

# 함수 실행 전에는 인스턴스 변수의 별도공간 존재 하지 않음
# 함수 실행 후, 인스턴스 변수 할당이 성사되면 생성