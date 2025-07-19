#Question 1

"""Create a Python program that simulates a student management system. Define four
classes: Person, Student, Teacher, and Admin. The Person class will serve as the base
class, while the Student, Teacher, and Admin classes will be derived from it using
hybrid inheritance.
1. Define the Person class with attributes for name, age, and gender. This class
should also have methods for displaying basic information about a person.
2. Define the Student class, which inherits from the Person class. The Student class
should have additional attributes for roll number and class, as well as methods
for displaying student-specific information.
3. Define the Teacher class, which also inherits from the Person class. The Teacher
class should have additional attributes for subject and experience, as well as
methods for displaying teacher-specific information.
4. Define the Admin class, which inherits from both the Student and Teacher
classes. The Admin class should have additional attributes for salary and
position, as well as methods for displaying admin-specific information.
5. Prompt the user to enter details for a student, a teacher, and an admin.
6. Create objects of the Student, Teacher, and Admin classes with the provided
details.
7. Display the information of all three objects using their respective display
methods."""

from tabulate import tabulate
class person:
    def __init__(self,name,age,gender):
        self.name=name
        self.age=age
        self.gender=gender
    def display(self):
        print(f"Name : {self.name}\nAge : {self.age}\nGender : {self.gender}")

class student(person):
    def __init__(self,name,age,gender,rollno,Class):
        person.__init__(self,name,age,gender)
        self.rollno=rollno
        self.Class=Class
    def display(self):
        table = [["Name", "Age", "Gender", "Roll No", "Class"],[self.name, self.age, self.gender, self.rollno, self.Class]]
        print(tabulate(table, headers="firstrow"))

class teacher(person):
    def __init__(self,name,age,gender,subject,exp):
        person.__init__(self,name,age,gender)
        self.subject=subject
        self.exp=exp
    def display(self):
        table = [["Name", "Age", "Gender", "Subject", "Experience"],[self.name, self.age, self.gender, self.subject, self.exp]]
        print(tabulate(table, headers="firstrow"))

class admin(student, teacher):
    def __init__(self, name, age, gender,subject, exp, rollno, Class,salary, position):
        teacher.__init__(self,name,age,gender, subject, exp)
        student.__init__(self,name,age,gender,rollno, Class)
        self.salary = salary
        self.position = position

    def display(self):
        teacher.display(self)
        student.display(self)
        table = [["Name", "Age", "Gender", "Subject", "Experience","Roll No","Class","Salary","Position"],[self.name, self.age, self.gender, self.subject, self.exp, self.rollno, self.Class, self.salary, self.position]]
        print(tabulate(table, headers="firstrow"))

while True:
    inp = int(input("Enter 1 for student, 2 for teacher, 3 for admin, 0 to exit: "))
    match inp:
        case 0:
            print("Thank You")
            break        
        case 1:
            name = input("Enter name: ")
            age = int(input("Enter age: "))
            gender = input("Enter gender: ")
            rollno = int(input("Enter rollno: "))
            Class = input("Enter class: ")
            s = student(name, age, gender,rollno, Class,)
            s.display()
        case 2:
            name = input("Enter name: ")
            age = int(input("Enter age: "))
            gender = input("Enter gender: ")
            subject = input("Enter subject: ")
            exp = int(input("Enter experience: "))
            t = teacher(name, age, gender,subject, exp)
            t.display()
        case 3:
            name = input("Enter name: ")
            age = int(input("Enter age: "))
            gender = input("Enter gender: ")
            rollno = int(input("Enter rollno: "))
            Class = input("Enter class: ")
            subject = input("Enter subject: ")
            exp = int(input("Enter experience: "))
            salary = int(input("Enter salary: "))
            position = input("Enter position: ")
            a = admin(name, age, gender, subject, exp, rollno, Class, salary, position)
            a.display()
        case _:
            print("Wrong Input\n")
            
            
            
#QUESTION 2            

"""The objective of this lab task is to create a class hierarchy that represents different
types of bank accounts and implement basic banking operations. This will help students
learn the fundamental concepts of object-oriented programming (OOP) in Python.
Tasks:
1. Create a base class 'Account':
This class will represent the basic structure of a bank account and define
common attributes and methods applicable to all types of accounts.
a. Attributes:
o account_number: A unique identifier for the account.
o balance: The current balance of the account.
o account_holder_name: The name of the account holder.
b. Methods:
o deposit(amount): Deposits the specified amount into the account balance.
o withdraw(amount): Withdraws the specified amount from the account
balance, ensuring it does not exceed the account balance.
2. Create a subclass 'SavingsAccount' that inherits from 'Account':
This class will represent a savings account, adding an attribute for the interest
rate and defining a method to calculate and add interest to the account balance.
a. Attribute:
o interest_rate: The annual interest rate applied to the account balance.
b. Method:
o calculate_interest(): Calculates the interest earned based on the
interest rate and adds it to the account balance.
3. Create a subclass 'CurrentAccount' that inherits from 'Account':
This class will represent a current account, adding an attribute for the overdraft
limit and modifying the withdrawal behavior to account for overdraft scenarios.
a. Attribute:
o overdraft_limit: The maximum amount that can be withdrawn beyond
the account balance.
b. Method:
o check_overdraft(amount): Checks if the specified withdrawal amount
exceeds the account balance and overdraft limit.
o withdraw(amount): Overrides the withdraw method from the base class to
check for overdraft and adjust the withdrawal accordingly.
4. Create instances of each class and perform banking operations:
Instantiating objects of the SavingsAccount and CurrentAccount classes and
calling their respective methods will demonstrate the implementation of objectoriented concepts and simulate real-world banking operations."""

from tabulate import tabulate
class Account:
    def __init__(self, name, num, bal):
        self.name=name
        self.num=num
        self.bal=bal
    def deposit(self,amount):
        if amount<0:
            print("Invalid Amount")
        else:
            self.bal+=amount
    def withdraw(self,amount):
        if self.bal<amount:
            print("Insufficient Balance")
        else:
            self.bal-=amount
    def display(self):
        table = [["Name", "Account Number", "Balance"],[self.name, self.num, self.bal]]
        print(tabulate(table, headers="firstrow"))

class savingsAccount(Account):
    def __init__(self, name,num,bal,interestRate):
        super().__init__(name,num,bal)
        self.interestRate=interestRate
    def interest(self):
        ir = self.bal*(self.interestRate/100)
        self.bal+=ir
        return self.bal
    def display(self):
        table = [["Name", "Account Number", "Interest Rate","Balance"],[self.name, self.num, self.interestRate, self.bal]]
        print(tabulate(table, headers="firstrow"))

class currentAccount(Account):
    def __init__(self,name,num,bal,overDraftLimit):
        super().__init__(name,num,bal)
        self.overDraftLimit = overDraftLimit
    def checkDraft(self,amount):
        if self.bal+self.overDraftLimit>=amount:
            return True
        else:
            return False
    def withdraw(self,amount):
        if self.checkDraft(amount):
            self.bal-=amount
        else:
            print("Limit Exceeds")
    def display(self):
        table = [["Name", "Account Number", "Over Draft Limit","Balance"],[self.name, self.num, self.overDraftLimit, self.bal]]
        print(tabulate(table, headers="firstrow"))

while True:
    inp = int(input("Enter 1 for saving account, 2 for current account, 3 for exit: "))
    match inp:
        case 1:
            name = input("Enter Name: ")
            num = int(input("Enter Account Number: "))
            bal = int(input("Enter Balance: "))
            rate = int(input("Enter Interest Rate: "))
            saving = savingsAccount(name,num,bal,rate)
            saving.interest()
            saving.display()
        case 2:
            name = input("Enter Name: ")
            num = int(input("Enter Account Number: "))
            bal = int(input("Enter Balance: "))
            dl = int(input("Enter Overdraft Limit: "))
            current = currentAccount(name,num,bal,dl)
            amount = int(input("Enter Amount to Withdraw: "))
            current.checkDraft(amount)
            current.withdraw(amount)
            current.display()
        case 3:
            print("Thank You")
            break
        case _:
            print("Invalid Input")
