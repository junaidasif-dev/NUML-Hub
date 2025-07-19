"""# updating list values
list = [1, 2, 3, 4, 5, 6]
print(list)
# It will assign value to the value to the second index
list[2] = 10
print(list)
# Adding multiple-element
list[1:3] = [89, 78]
print(list)
# It will add value at the end of the list
list[-1] = 25
print(list)
print(list[2])
list[1+3] = 9
print(list)"""


"""#Declaring the empty list
l =[]
#Number of elements will be entered by the user
n = int(input("Enter the number of elements in the list:"))
# for loop to take the input
for i in range(0,n):
# The input is taken from the user and added to the list as the item
    l.append(input("Enter the item:"))
print("printing the list items..")
# traversal loop to print the list items
for i in l:
    print(i, end = " ")"""


"""l = []
n = int(input("Enter the number of elements in the list:"))
for i in range(0,n):
    #append
    l.append(int(input("Enter integer:")))
print(l)
    #copy
l.copy()
print(l)
    #count
i = l.count(5)
print(i)
l1 = [0,7,9,6]
    #extend
l.extend(l1)
print(l)
    #index
print(l.index(0))
    #insert
l.insert(2,5)
print(l)
    #pop
l.pop(3)
print(l)
    #remove
l.remove(0)
print(l)
    #reverse
l.reverse()
print(l)
    #sort
print(sorted(l))"""


#Python program to find unique numbers in a given list.

"""list = [1,3,3,0,1,7,5,9,5,6]
unique = []
for n in list:
    if n not in unique:
        unique.append(n)
print("Unique numbers in the list:", unique)"""


#Python program to find sum of all numbers in a list.

"""list = []
sum = 0
n = int(input("Enter the number of elements in the list:"))
for i in range(0,n):
    list.append(input("Enter integer:"))
    sum = sum+(i+1)
print(sum)"""


#Python program to create a list of 5 random integers.

"""list = []
print("Enter 5 integers in the list")
for i in range(0,5):
    list.append(int(input("Enter integer:")))
print(list)"""


"""Write a program that creates a list of 10 random integers. Then create two
lists by name odd_list and even_list that have all odd and even values of the
list respectively."""

"""list = []
even_list = []
odd_list = []
print("Enter 10 integers in the list")
for n in range(0,10):
    list.append(int(input("Enter integer:")))
for n in list:
    if n % 2 == 0:
        even_list.append(n)
    else:
        odd_list.append(n)
print("List of even numbers\n",even_list,"\nList of odd numbers\n",odd_list)"""


#Python program to remove all odd numbers from a list.

"""list = []
print("Enter 10 integers in the list")
for n in range(0,10):
    list.append(int(input("Enter integer:")))
for n in list:
    if n % 2 == 0:
        list.append(n)
    else:
        list.remove(n)
print("List after removing odd numbers is look like: ",list)"""


#Python program to sort a list of strings on the number of alphabets in each word.

"""input_list = ["apple", "banana", "cherry", "date", "fig"]
sorted_list = []
while input_list:
    min_length = len(input_list[0])
    min_word = input_list[0]
    for word in input_list:
        if len(word) < min_length:
            min_length = len(word)
            min_word = word
    sorted_list.append(min_word)
    input_list.remove(min_word)
print("Sorted list by the number of alphabets:", sorted_list)"""


#Python program non-numeric items in a list in a separate list.

"""list = [1, 'apple', 2.5, 'banana', 3, 'cherry', 'date', 4]
non_numeric_list = []
for item in list:
    if type(item) is str:
        non_numeric_list.append(item)
print("Non-Numeric Items:", non_numeric_list)"""


#Python program to create a list of integers representing each character in a string

"""s = 'Junaid'
list = []
for n in s:
    list.append(n)
print(list)"""


#Python program to find numbers common in two lists.

"""l1 = [1,2,3,5,6,7]
l2 = [2,10,2,5,0,9]
l3 = []
for n in l1:
    for i in l2:
        if n == i:
            l3.append(n)
            break
print(l3)"""