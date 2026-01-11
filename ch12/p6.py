path = r"python\ch12\order.txt"
with open(path, 'a', encoding="utf-8") as order:
    order.write(input("주문내역:\n"),end="\n")

with open(path, 'r', encoding="utf-8") as order:
    order.read()