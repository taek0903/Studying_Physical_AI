# slicing.py

# 슬라이싱
week = ["월", "화", "수", "목", "금", "토", "일"]
print(week)

print(week[2:5])
print(week)
print(type(week))
print(type(week[2:5]))

# 문제1: 토/일 데이터를 출력하시오.
print(week[5:7])

# 문제2: 월~목 데이터를 출력하시오.
print(week[0:4])

print("음수 인덱싱 -------------------------")
print(week[-1])       # 뒤에서 부터
print(week[-2])
print(week[-3])

print("음수 슬라이싱 -------------------------")
print(week[-3:-1])
print(week[-1:4]) # 슬라이싱 방향 좌=>우
print(week[-1:1])

print("슬라이싱 -------------------------")
print(week[0:5])
print(week[0:4+1])

print("인덱스 번호 생략하는 경우 -------------------------")
# 1. 시작 인덱스 생략
print(week[:5])

# 2. 끝 인덱스 생략 => 맨 마지막 데이터를 가지고 오겠다.
print(week[0:])

# 3. 모든 인덱스 생략 => 전체 데이터 대상
print(week[:])

print("-------------------------")
print(week[0:6:2]) #[0(시작):+1(끝):스탭]
print(week[0:7:3])