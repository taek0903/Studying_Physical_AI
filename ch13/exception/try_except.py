# try_except.py

# try:
#     코드블록
# except 예외클래스:
#     코드블록

# try:
#     코드블록
# except 예외클래스:
#     코드블록
# fianlly:
#     코드블록

while True:
    try:
        x = int(input("Please enter a number: "))   # ValueError
        result = 10*(1/x)        # ZeroDivisionError
        print(result)
        break
    except ValueError as e: 
        print("Oops! That was no valid number. Try again...")
        print(e)
    # except ZeroDivisionError as err:
    #     print("0으로 나눌수 없습니다.")
    #     print(err)
    except Exception as e:
        print("예외 발생!")
        print(e)


