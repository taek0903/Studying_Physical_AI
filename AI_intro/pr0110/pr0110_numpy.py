import numpy as np

# numpy 부동소수점 자리 표시
np.set_printoptions(suppress=True, precision=3)

np.array([1,2,3,4,5,6])
print(np.array([1, 2, 3, 4, 5, 6]))
print('---------------------------')

n1=np.array([1 ,2, 3, 4, 5, 6])
print(n1)
print('----------------------------')

# 오소수 확인
print(n1.shape) # 1차원
print('----------------------------')

len(n1)
print(n1)
print('----------------------------')

n2 = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [10,11,12]
])
print(n2)
print(n2.shape)
print('----------------------------')

# zeros, ones
n3=np.zeros(5)
print(n3, n3.shape) # 1개의 배열에 5개의 원소가 있음 (5,)   1차원
print('----------------------------')

n4=np.ones((2,3))   # 2개의 배열에 각각의 원소가 3개 있음   2차원 (Cannel, Hight, Width)
print(n4)
print(n4.shape)
print('----------------------------')

n5 = np.random.randn(2,3,4) # 2개의 채널에 3개의 배열이 안에 각각 원소 4개가 있음   3차원 
print(n5)
print(n5.shape)
print('----------------------------')

# 그래프 출력용 수치 배열 생성
n6 = np.linspace(-1,1,11)
print(n6)
print(n6.shape)
print(n6.ndim)
print('----------------------------')

# 조작(manupulatiuon)
print(n2)
print(n2.shape)
print('----------------------------')

print(n2[:,0])  # 1열을 추출
print(n2[0,:])  # 1행을 추출
print('----------------------------')

n7=np.array([False, True, True, False])
print(n7.shape)
n8=n2[n7]   # 마스킹 (2행 3행 추출)
print(n8)
print(n8.shape)
print('----------------------------')

# reshape(** 중요 **)
n9=np.array(range(24))
print(n9.shape)
print('----------------------------')

print(n9.reshape(3,-1))
print(n9.reshape(3,8))
print(n9.reshape(2,2,-1))
print(n9.reshape(2,2,2,-1)) # -1을 지정 => 자동 계산
print('----------------------------')

# 축 교체하기
print(n2)
print(n2.T)
print('----------------------------')

n10 = n9.reshape(2,-1,4)
print(n10)
print(n10.shape)
print('----------------------------')

print(np.transpose(n10))
print(np.transpose(n10).shape)
print(np.transpose(n10,(1,2,0)))
print(np.transpose(n10,(1,2,0)).shape)
print('----------------------------')

# 행렬 연결하기
n13 = np.array(range(1,7)).reshape(2,3)
n14 = np.array(range(7,13)).reshape(2,3)
print(n13)
print(n14)
print('----------------------------')

n15 = np.array(range(14,17))
print(n15)
print('----------------------------')

print(np.vstack([n13, n14]))
print(np.hstack([n13, n14]))
print('----------------------------')

print(np.concatenate([n13, n14]))
print(np.concatenate([n13, n14], axis=0))   # axis = 0 default값 hstack이랑 유사
print(np.concatenate([n13, n14], axis=1))   # axis = 1 hstack이랑 유사
print('----------------------------')

# 연산
print(n13)
print(n14)
print('----------------------------')

n15 = n13 + n14
print(n15, n15.shape)
print('----------------------------')

# 브로드캐스팅
print(n1)
print(n1-3) # n1의 모든 값에서 3의 값을 빼고 싶어요
print('----------------------------')

# 유니버셜 함수
print(np.linspace(0,10,5))  # 0부터 10까지 5개의 원소를 배열
x = np. linspace(0,2*np.pi, 5)  # 0부터 2*pi까지 5개의 원소 배열
print(x)
y = np.sin(x)   # sin()에 x값 대입
print(y)
print('----------------------------')

# 집계함수
print(n1)
print(np.sum(n1))   # int
print(np.mean(n1))  # float
print(np.max(n1))   # int
print(np.min(n1))   # int
print('----------------------------')

# 응용예시

# 정답
yt = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

# 예측
yp = np.array([1, 1, 0, 1, 0, 1, 0, 1, 1, 1])

# 정답(yt) 예측(yp) 비교
matched = (yt==yp)
print(matched)
print(matched.sum())
print('----------------------------')

correct = matched.sum()
total = len(matched)
accuracy = correct / total
print(accuracy)
print(f'정답수: {correct}, 전체 개수: {total}, 정확도: {accuracy*100}%')
print('----------------------------')

n1_max = n1.max()
n1_min = n1.min()
print(n1_max, n1_min)

print((n1 - n1_min) / (n1_max - n1_min))
print('----------------------------')