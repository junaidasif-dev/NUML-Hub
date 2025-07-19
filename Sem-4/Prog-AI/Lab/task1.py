#Write a program that checks if a number entered by the user is positive, negative, or zero.

"""n = int(input("Enter number:"))
if n<0:
    print(f"{n} is negative number")
elif n>0:
    print(f"{n} is positive number")
else:
    print(f"{n} is  zero")"""
    
    
#Create a program that determines if a given year is a leap year.

"""n = int(input("Enter year:"))
if n%4 == 0:
    print(f"{n} is a leap year")
else:
    print(f"{n} is not a leap year") """
    
    
#Implement a program that calculates the factorial of a number using a while loop.

"""n = int(input("Enter number:"))
f=1
while(n>0):
    f = f*n
    n = n-1
print(f"Factorial = {f}")"""


#Write a program that checks if a given number is prime or not.

"""number = int(input("Enter a number: "))

if number <= 1:
  print(number, "is not a prime number")
elif number <= 3:
  print(number, "is a prime number")
elif number % 2 == 0 or number % 3 == 0:
  print(number, "is not a prime number")
else:
  i = 5
  while i * i <= number:
    if number % i == 0 or number % (i + 2) == 0:
      print(number, "is not a prime number")
      break
    i += 6
  else:
    print(number, "is a prime number")"""


#Create a program that prints the Fibonacci series up to a specified number of terms.

"""n = int(input("Enter term:"))
a = 0 
b = 1
if n<0:
    print("Invalid input")
else: 
    c=0
    while(c<n):
        print(a)
        d = a+b
        a = b
        b = d
        c = c+1"""
  
  
 #Implement a program that calculates the sum of all numbers in a given range using a for loop. 
 
"""n = int(input("Enter number:"))
sum = 0
for i in range(1,n+1):
    sum = sum+i
print(sum)"""

#Write a program that checks if a given string is a palindrome.

"""n = input("ENTER  STRING\n")
length = int(len(n))

for i in range(0, int(length/2 + 1)):
   if n[i] == n[-i - 1]:
      i += 1
   else:
      break

if i < (length / 2):
   print("not")
else:
   print("yes")"""
   
   
#Create a program that generates the multiplication table of a given number using nested loops.

"""num = int(input("Enter a Numbwe: "))

for i in range(1,num+1):
    for j in range(1, 11):
        print(i, "x", j, "=", i * j, "\n")

    print("\n")
    """
    
#Implement a program that finds the largest element in an array using if-else statements.

"""heights = [100, 2, 300, 10, 11, 1000]

largest_number = heights[0]

for number in heights:
    if number > largest_number:
        largest_number = number

print(largest_number)"""


#Write a program that simulates a simple calculator with options for addition, subtraction, multiplication, and division.

"""print("Welcome to Simple Calculator!")

while True:
    print("\nOptions:")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Division")
    print("5. Exit")

    choice = input("Enter your choice (1/2/3/4/5): ")

    if choice == '5':
        print("Exiting the calculator.")
        break
    elif choice in ('1', '2', '3', '4'):
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))

        if choice == '1':
            print("Result:", num1 + num2)
        elif choice == '2':
            print("Result:", num1 - num2)
        elif choice == '3':
            print("Result:", num1 * num2)
        elif choice == '4':
            if num2 == 0:
                print("Error! Division by zero is not allowed.")
            else:
                print("Result:", num1 / num2)
    else:
        print("Invalid input! Please enter a valid option.")"""