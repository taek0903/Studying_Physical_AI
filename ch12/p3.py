# p3.py

path=r"python\ch12\file3.txt"
with open(path, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    account = {}
    for item in lines:
        # name, num = item.split()
        lineList=item.split()
        account[lineList[0].strip()]=lineList[1].strip()
    print(account)

