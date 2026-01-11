# 1번
movie_rank=['하얼빈', '무파사_라이온킹', '소방관']
print(movie_rank)
print("-----------------------------------------")
# 2번
movie_rank.append('위키드')
movie_rank.append('해리포터')
movie_rank.insert(0,'반지의 재왕')
print(movie_rank)
print("-----------------------------------------")
# 3번
movie_rank.insert(3,'모아나2')
print(movie_rank)
print("-----------------------------------------")
# 4번
movie_rank.remove('소방관')
del movie_rank[5]
print(movie_rank)
print("-----------------------------------------")
# 5번
nums=[1, 2, 3, 4, 5]
total= sum(nums[:])
print(total)
print("-----------------------------------------")
# 6번
cook=['피자','김밥','만두','양념치킨','족발','피자','김치만두','쫄면','쏘세지','라면','팥빙수','김치전']
total_count = len(cook)
print(total_count)
print("-----------------------------------------")
# 7번
t = 1,2,3,4
print(type(t))
print("-----------------------------------------")
# 8번
t = ('a','b','c')
listt = list(t)
listt[0] = 'A'
t = tuple(listt)
print(t, type(t))
print("-----------------------------------------")
# 9번
ice = {"메로나":1000, "폴라포":1200, "빵빠레":1800}
print(ice)
print("-----------------------------------------")
# 10번
ice["조스바"] = 1200
ice["월드콘"] = 1500
print(ice)
print("-----------------------------------------")
# 11번
ice['메로나']=1300
print(ice)