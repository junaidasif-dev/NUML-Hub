class Employee:
    def calculateSalary(self):
        return 0

class Manager(Employee):
    def __init__(self,salary,bonusPercentage):
        self.salary=salary
        self.bonusPercentage=bonusPercentage
    def calculateSalary(self):
        self.salary = self.salary + (self.salary*(self.bonusPercentage/100))
        return self.salary

class Developer(Employee):
    def __init__(self, wage, hours):
        self.wage=wage
        self.hours=hours
    def calculateSalary(self):
        return self.wage*self.hours

class Designer(Employee):
    def __init__(self, salary):
        self.salary=salary
    def calculateSalary(self):
        return self.salary
    

while True:
    inp = int(input("1. Employee\n2. Manager\n3. Developer\n4. Designer\n5. Exit\n"))
    match inp:
        case 1:
            emp = Employee()
            print("\nDisplaying Employee Details\n")
            print("Employee Salary: ",emp.calculateSalary())
            print("\n")
        case 2:
            salary = int(input("Enter Manager Base Salary: "))
            bonus = int(input("Enter Manager Bonus Percentage: "))
            man = Manager(salary,bonus)
            print("\nDisplaying Manager Details\n")
            print("Manager Salary: ",man.calculateSalary())
            print("\n")
        case 3:
            wage = int(input("Enter Developer Wage Per Hour: "))
            hour = int(input("Enter Developer Hours Worked: "))
            dev = Developer(wage,hour)
            print("\nDisplaying Developer Details\n")
            print("Developer Salary: ",dev.calculateSalary())
            print("\n")
        case 4:
            salary = int(input("Enter Designer Salary: "))
            des = Designer(salary)
            print("\nDisplaying Designer Details\n")
            print("Designer Salary: ",des.calculateSalary())
            print("\n")
        case 5:
            print("\n")
            print("Thank You")
            break
        case _:
            print("\n")
            print("Invalid Input")
            print("\n")