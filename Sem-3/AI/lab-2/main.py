
"""for i in range(10,50,5):
    print(i)"""


'''def mby2(x):
    return x*2
a = mby2(70)
print(a)

print (mby2(3))
'''

# 1. Factorial

'''print("Enter number to find a factorial:")
num = int(input())
fac=1
for i in range(1,num+1):
    fac = fac*i
print("Factorial of ", num, " is ", fac)'''

# 2. Cube

'''print("Enter number:")
num = int(input())
for i in range(1,num+1):
    a = i*i*i
    print("Cube of ", i, " is ",a )'''

# 3. Table

'''print("Enter number:")
num = int(input())
for i in range(1,11):
    a = num*i
    print(num," X ", i, " = ",a )'''

# 4. Sum

'''print("Enter number:")
num = int(input())
s=0
for i in range(1,num+1):
    s = s+i
print("Sum = ", s)'''

# 5. Present/Absent

"""def attendence(roll_no):
    roll_nos = [1, 2, 3, 4, 5, 6, 7]
    if roll_no in roll_nos:
        print(roll_no, " is present")
    else:
        print(roll_no, " is absent")

rn = int(input("Enter roll no: "))
attendence(rn)"""

# 6. Maximum

'''def num(a,b,c):
    print(max(a,b,c))

num(3,19,5)
'''

# 7. Area of Circle

'''def area(r):
    area = 3.14*r*r
    return area
print("Area of circle is ",area(3))'''

# 8. Largest Number

"""def large_num():
    print("Largest number is ", max(numbers))

numbers = []
print("\tEnter any 10 numbers\t")
for i in range(10):
    num = int(input())
    numbers.append(num)
large_num()"""
