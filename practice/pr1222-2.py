# 1번
print("1--------------------")
a=11 
b=5
print(a+b)
print(a-b)
print(a*b)
print(a//b)
print(a%b)

# 2번
print("2--------------------")
a=90
if a >=90:
    print("A")
elif a >=80:
    print("B")
elif a >=70:
    print("C")
else:
    print("D")

# 3번
print("3--------------------")
c=[2,8,6,5,7,9,1,4]
d=[]
e=[]
for i in c:
    if i % 2 ==0:
        d.append(i)
    else:
        e.append(i)
print(d,e)

# 4번
print("4--------------------")
def range_sum(a,b):
    start=min(a,b)
    end=max(a,b)
    total=0
    for i in range(start,end+1):
        total += i
    print(total)
range_sum(10,5)

# 5번
def alphabet_freq(text):
    freq_dict={}

    for char in text.lower():
        if char.isalpha():
            freq_dict[char]=freq_dict.get(char, 0)+1
    
    if not freq_dict:
        return None, 0
    
    most_common_char = max(freq_dict, key=freq_dict.get)
    return most_common_char, freq_dict[most_common_char]

text = "Apple Banna Orange"
char, count = alphabet_freq(text)
print(f"결과: {char} ({count}개)")

# def alphabet_freq(text):
#     freq_dic={}
#     for char in text.lower:
#         freq_dic[char]=freq_dic.get(char,0)+1
    
#     if not freq_dic:
#         return None, 0
    
#     most_common_char = max(freq_dic, key=freq_dic.get)
#     return most_common_char, freq_dict[most_common_char]