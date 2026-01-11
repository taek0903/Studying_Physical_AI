# hw19.py

# print('6-------------------')
# import statsmodels.api as sm
# import numpy as np

# X = [1,2,3,4,5]
# y = [2,4,6,8,10]

# X = sm.add_constant(X)
# model = sm.OLS(y,X).fit()

# print(f'절편(b0): {model.params[0]}')
# print(f'기울기(b1): {model.params[1]}')

# print('7-------------------')
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt

# x = [1,2,3,4,5]
# y = [2,4,6,8,10]

# X = sm.add_constant(x)
# model = sm.OLS(y,X).fit()

# y_pred = model.predict(X)

# plt.scatter(x, y, label="Data")
# plt.plot(x,y_pred, label="Regression")

# plt.title('Linear Regression')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

# print('8-------------------')
# from sklearn.linear_model import LogisticRegression

# X = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
# y = [0,0,0,0,1,1,1,1,1,1]

# model = LogisticRegression()
# model.fit(X,y)

# pred = model.predict([[4.5],[6.5]])
# print(pred)

# print('9-------------------')
# from scipy.optimize import root
# def equation(x):
#     return (x-3)**2

# sol = root(equation, x0=1)
# print(f'Root: {sol.x}')

# print('10------------------')
# from scipy.stats import describe

# group_A = [80, 85, 90, 75, 95]
# stats = describe(group_A)
# print(stats)





