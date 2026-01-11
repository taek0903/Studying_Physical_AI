path = r"python\ch12\order.txt"
with open(path, 'w', encoding="utf-8") as order:
    order.write(input("주문내역:\n"))

with open(path, 'r', encoding="utf-8") as order:
    order.read()