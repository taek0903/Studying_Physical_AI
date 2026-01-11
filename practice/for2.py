# 2중 for문
# 2차원 리스트 전체 순회하기
grid = [
    [1, 2, 3],
    [4, 5, 6]
]

for row in grid:
    for value in row:
        print(value)
    print()
print('------------')
# 구구단
for dan in range(2,10):
    for i in range(1,9):
        print(f"{dan} X {i} = {dan*i}")
    print('------------')
# 별찍기
for i in range(1,6):
    for j in range(i):
        print("*", end="")
    print()
print('------------')
# 직사각형 테두리 그리기
rows, cols = 5, 10
for i in range(rows):
    for j in range(cols):
        if i == 0 or i == rows -1 or j == 0 or j == cols-1:
            print("#", end="")
        else:
            print(" ", end="")
    print()
print('------------')
# 1~9까지 "곱셈 표(매트릭스)" 만들기
for i in range(1,10):
    for j in range(1,10):
        print(f"{i*j:2}", end="")
    print()
print('------------')
# 좌표찍기(x,y) 조합 전부 출력
for y in range(3):
    for x in range(4):
        print(f"({x}, {y})", end="")
    print()
print('------------')