# default1.py

# 디폴트 매개변수
# def 함수명(매개변수1=값1, 매개변수=값2 ...):
#     코드블록
#     return 반환값

# def persona(width, height):
def persona(width=11, height=21):
    print(f"함수디폴트값없음=>있음 \nwidth= {width} height= {height}")

# def persona():
#     print(f"매개변수 없는 함수")

def personb(width=4, height=3): # 인수가 없으면 그대로 사용하세요~
    print(f"함수디폴트값설정 \nwidth= {width} height= {height}")

persona(10,20)
persona()
personb(30,40)
personb()

print('------------------------------------')
# def persona(width=11, height):
def persona(width, height=11): 
    print(f"함수디폴트값없음=>있음 \nwidth= {width} height= {height}")

def personb(width=4, height=3):
    print(f"함수디폴트값설정 \nwidth= {width} height= {height}")

# persona(10,20)
persona(30)
personb(30,40)
personb()

# 파이썬에서 항수 정의시 기본값이 있는 매개변수는
# 기본값이 없는 매개변수 뒤에 위치한다.
print('------------------------------------')
def personc(age=29, weight=80, hight=182):
    print("함수기본값설정")
    print(f"나이: {age} 몸무게: {weight} 키: {hight}")

personc(25, 78, 182)
personc()

# 1. 모든 매개변수에 기본값 설정가능
# 2. 부분 매개변수에 기본값 설정시 뒤에서부터 설정할 것
# 3. 기본값이 있더라도 인수를 전달 가능(전달되는 인수가 우선 처리)

personc(33, 55, hight=168) # => 키워드 인자, 위치 인자
personc(33, hight=55, weight=168)

def add(x: int, y: int):
    return x+y
print(add(3,5))
print(add(3.2,5))