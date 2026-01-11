# func2.py

def fplusminus(arg):
    if arg > 0:
        return "plus", arg
    elif arg < 0:
        return "minus", arg
    else :
        return "zero", arg
stra, num = fplusminus(0)
print(stra, num)
stra, num = fplusminus(-3)
print(stra, num)
stra, num = fplusminus(14)
print(stra, num)

print('----------------------')
def mayabs(arg): # 절댓값을 반환해주는 기능 수행
    if(arg<0):
        result= arg*-1
    else:
        result = arg
    return result
print(mayabs(10))
print(mayabs(-9))
print(mayabs(0))