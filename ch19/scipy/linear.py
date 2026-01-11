# linear.py
# import scipy as sp

# 선형 방정식 해 구하기
# from scipy.linalg import solve

# A = [[3,1],[1,2]]
# b = [9,8]

# x= solve(A,b)
# print(f'Solution:{x}')

# from scipy.optimize import minimize

# def f(x):
#     return x**2 + 4*x + 4

# result = minimize(f, x0=0)
# print(f'Optimal Value: {result.x}')

# find_root.py
# from scipy.optimize import root
# def equation(x):
#     return x**2-4

# sol = root(equation, x0=1)
# print(f"root: {sol.x}")

# # descirbe.py
# from scipy.stats import describe

# data=[1,2,3,4,5,6,7]
# stats = describe(data)
# print(stats)

# 외도(Skewness)는 분포의 비대칭 정도
# 첨도(Kurtosis)는 분포의 뾰족한 정동

# sparse_matrix.py
from scipy.sparse import csr_matrix

data = [
    [0,0,3],
    [4,0,0],
    [0,5,0]
]
sparse_matrix = csr_matrix(data)
print(sparse_matrix)