# pr1218.py

class Phone:
    def __init__(self, maker, year, color):
        self.maker=maker
        self.year=year
        self.color=color
    def info(self):
        print(self.maker, self.year, self.color)

my_phone=Phone('samsung', 2025, 'black')
my_phone.info()
print('---------------------------')

class Phone:
    def __init__(self, maker, year, color):
        self.maker=maker
        self.year=year
        self.color=color
    def setinfo(self, maker, year, color):
        self.maker=maker
        self.year=year
        self.color=color
    def info(self):
        print(self.maker, self.year, self.color)

my_phone = Phone("samsung", 2025, "black")
my_phone.info()
my_phone.setinfo('apple',2025,'white')
my_phone.info()
print('---------------------------')

class Animal:
    def __init__(self):
        self.name="고양이"
    def sound(self):
        print("야옹")
cat=Animal()
print(cat.name)
cat.sound()
print('---------------------------')

class BankAccount:
    def __init__(self, balance):
        self.balance=balance
        print(self.balance,'원')
    def time(self, n):
        self.n=n
        print(self.n,"번")
    def d(self, x):
        self.x=x
        self.balance += self.x
        print(self.balance,"원")
    def w(self, x):
        self.x=x
        if self.balance >= self.x:
            self.balance -= self.x
            print("OK")
            print(self.balance,"원")
        else:
            print("Fail")
            print(self.balance,"원")

my_account = BankAccount(1000000)
my_account.time(2)
my_account.d(1000000)
my_account.w(3000000)
print('---------------------------')

class Student:
    def __init__(self, name, scores):
        self.name = name
        self.scores = scores
    def avg(self):
        return sum(self.scores)/len(self.scores)
st=Student("우택",[100,80,90,80])
print(f"{st.name} 평균: {st.avg():.2f}")