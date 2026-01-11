# Bag.py

# 가방 클래스 정의
class Bag:
    # 클래스 맴버변수
    # 맴버함수(매서드)
    def __init__(self, name):  # 초기화 매서드
        # 인스턴스 맴버변수
        self.data = []   # 가방에 추가될 물건 종류
        # self.name = "backpack"
        self.name = name
    
    # 기능 : 담다(추가하다)
    # 입력 : 물건 => x
    def add(self, x):
        print("추가하다")
        self.data.append(x)
    
    def addtwice(self, x):
        print("2번 추가하다")
        self.add(x)
        self.add(x)

# 객체: 배낭, 크로스백, 케리어
backpack = Bag("backpack")    # 초기값 설정
print("배낭이름:", backpack.name)
print("가방에 담긴 물건",backpack.data)
backpack.add("휴대폰")
backpack.add("지갑")
backpack.add("책")
# print("가방에 담긴 물건",backpack.data)
backpack.addtwice("돈")
print("가방에 담긴 물건",backpack.data)

print(backpack.name)
# globals() # 전역 변수명 확인
# for name, obj in list(globals().items()):
#     if obj is backpack:
#         print(name)


# # 원하는 형태의 가방 객체를 하나 생성하고 활용해 보세요.
crossbag = Bag("crossbag")
print("핸드백이름:", crossbag.name)
crossbag.add("지갑")
crossbag.addtwice("보조배터리")
print("가방에 담긴 물건", crossbag.data)
