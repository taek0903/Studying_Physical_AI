# dict1.py

my_dict1 = {}
print(my_dict1)

my_dict2 = {0:1, 1:-2, 2:3.14}
print(my_dict2)

my_dict3 = {'이름':'엘리스', '나이':'10', '시력':[1.0, 1.2]}
print(my_dict3)
print(type(my_dict3))

# 값 접근하기
# 딕셔너리명['키']
print(my_dict3['이름'])
print(my_dict3['나이'])
print(my_dict3['시력'])

clover ={'나이':27, '직업':'병사'}
print(clover)
clover['번호'] = 9
print(clover)

clover['번호'] = 6
print(clover)
print("번호1:", clover['번호'])
print("번호2:", clover.get('번호'))
print(clover.get('직업'))


print("----------------------------")
# 이름, 나이, 성별, 원하는 데이터 1가지
my_dict4 = {"이름":"임진택", "나이": 29, "성별":"남자", "취미":['게임', '농구']}
print(my_dict4)
# 생성된 딕셔너리에 전화번호와 주소를 추가하시오
my_dict4['전화번호'] = '010-1234-5678'
my_dict4['주소'] = '광주시'
print(my_dict4)

# 12월31일이 지났습니다. 나이를 +1 하여 수정하시오.
my_dict4['나이'] = my_dict4['나이']+1
print(my_dict4['나이'])