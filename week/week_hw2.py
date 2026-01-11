print("1--------------------")
def solution(a,b):
    sum = a+b
    sub = a-b
    multi = a*b
    return sum, sub, multi

print("2--------------------")
def plus(a):
    total = 0
    for i in range(1,a+1):
        total += i
    return total
print(plus(10))

print("3--------------------")
def example1():
    x =20
    print(x)

example1()

class Student:
    school = "High School"

    def __init__(self,name,grade):
        self.name=name
        self.grade=grade

s1=Student("Alice",1)

print(Student.school)
print(s1.school)
print(s1.name)
print("6--------------------")
def min(num_lst):
    num_min = num_lst[0]
    for i in num_lst:
        if i < num_min:
            num_min = i
    print(num_min)

min([45,17,23,56,9,34])
print("9--------------------")
from math import factorial
result = factorial(5)
print(result)

print("10-------------------")
class Animal:
    def speak(self):
        return "animal speaks"
    
class Dog(Animal):
    def speak(self):
        return "Woof"

dog = Dog()
print(dog.speak())