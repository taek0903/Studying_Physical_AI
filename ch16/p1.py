# p1.py

# 1. 빈스텍 생성
# 2. 문자열 내 문자를 가져와
# 3. 만약, '시작' 괄호가 있다면 스택에 push
# 4. '종료' 괄호가 있다면 스택에서 pop
#   4-1. 만약, 스택이 비어있다면(매칭되는 시작괄호 없음) => False
#   4-2. pop한 괄호를 확인했는데 서로 일치하지 않으면 => False
# 5. 스택이 비어있다면 True(짝맞음)
def check_brackets(text):
    stack=[]                            # 스텍 생성
    pairs={')':'(', '}':'{', ']':'['}
    for char in text:                   # 문자 확인
        if char in '({[':               # 시작 문자 확인
            stack.append(char)          # 스택 추가
            print(f'stack={stack}')
        elif char in ']})':             # 종료 문자 확인    
            if not stack:
                return False
            top = stack.pop()           # 스택 제거
            print(top)
            print(f'char={char}')
            print(f'{pairs[char] != {top}}')
            if pairs[char] !=top:
                return False
    return len(stack) == 0

print(check_brackets("(a+b)"))
print('-------------------')
print(check_brackets("(a+b)}"))
print('-------------------')
print(check_brackets("[{(x+y)+3}-4]"))
print('-------------------')
print(check_brackets(")a+b"))