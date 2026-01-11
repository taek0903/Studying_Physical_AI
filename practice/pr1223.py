# pr1223.py

# 6번
def odd_even_swap(lst):
    last_odd_idx = None
    first_even_idx = None
    for i in range(len(lst)):
        if lst[i] % 2 == 0:
            first_even_idx = i
            break
    for j in range(len(lst)-1,-1,-1):
        if lst[j] % 2 != 0:
            last_odd_idx = j
            break
    lst[first_even_idx], lst[last_odd_idx]=lst[last_odd_idx], lst[first_even_idx]
    return lst

data = [1,8,9,7,5,3,2,4,6]
print(odd_even_swap(data))

# 7번
def sum_values(d):
    data = d.values()
    total = 0
    for i in data:
        total += i
    return total
data={1:10, 2:20, 3:30, 4:40}
print(sum_values(data))

# 8번
n=5
for i in range(1,n+1):
    for j in range(1,i+1):
        print(j,end="")
    print()

# dic1
def swap_first_even_last_odd(nums):
    odd_idx=None
    even_idx=None
    for i in range(len(nums)):
        if nums[i] % 2 == 0:
            even_idx=i
            break
    for j in range(len(nums)-1,-1,-1):
        if nums[j] % 2 != 0:
            odd_idx = j
            break
    temp = nums[even_idx]
    nums[even_idx] = nums[odd_idx]
    nums[odd_idx] = temp
    return nums
print(swap_first_even_last_odd([1,2,3,4,5,6,7,8,9]))

# dic2
def max_with_index(nums):
    max_idx=0
    for i in range(1,len(nums)):
        if nums[max_idx] < nums[i]:
            max_idx = i
    data = (nums[max_idx],max_idx)
    return data
data=[10,17,13,11]
print(max_with_index(data))

# dic3
def unique_keep_order(items):
    seen = set()                    # 중복되지 않는 자료형에 들어감
    result = []                     # 결과값(빈리스트)
    for x in items:                 # items의 갯수 만큼 반복
        if x not in seen:           # seen의 자료에 없는 자료가 들어오면
            seen.add(x)             # set 자료형에 추가
            result.append(x)        # 결과값 리스트에 추가
    return result

# dic4
def mean_point(points):
    if not points:                  # 빈 리스트 반환
        return None
    
    sum_x=0                         # x 좌표의 총합
    sum_y=0                         # y 좌표의 총합
    for (x,y) in points:            # 튜플 반복
        sum_x += x                  # x 좌표 더해주기
        sum_y += y                  # y 좌표 더해주기

    n = len(points)                 # 각 좌표의 갯수
    return(sum_x / n, sum_y / n)    # 각 좌표의 평균    

points=[(0,0), (2,2), (4,2)]
print(mean_point(points))

# dic5
data={"a":10, "b":20, "c":30}
def sum_and_avg(scores):
    data = scores.values()
    sum = 0
    avg = 0
    for i in data:
        sum +=i
    avg = sum / len(scores.values())
    return sum, avg
print(sum_and_avg(data))

# dic6
def key_of_max_value(d):
    if not d:
        return None
    
    best_key=None
    best_val=None

    for k, v in d.items():      # d.items 를 사용하면 key값과 value값이 튜플 형태로 나옴
        if best_val is None or v > best_val: # best_key, best_val의 초기값의 None이기 때문
            best_val = v
            best_key = k
    return best_key
data={"a":10, "b":20, "c":30}
print(key_of_max_value(data))

# dic7
def group_by_value(d):
    dic={}
    for k,v in d.items():
        if v not in dic:    # dic 데이터 공간에 v라는 키값이 없을 경우
            dic[v]=[]       # 새로운 키를 만들고 리스트 자료 공간을 준비해 둔다
        dic[v].append(k)
    return dic
fruit_color = {"apple":"red", "banana":"yellow", "cherry":"red"}
print(group_by_value(fruit_color))