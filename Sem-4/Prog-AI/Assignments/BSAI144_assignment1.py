"""5.1.Create a Python function that prompts the user to enter the size of a list and then, 
based on that size, allows the user to input list items. After collecting the specified 
number of items, the function should return the generated list."""

def create_list(size):
    list = []
    for i in range(size):
        list.append(input("Enter the list item: "))
    return list
size = int(input("Enter the size of the list: "))
final_list = create_list(size)
print(final_list)


"""5.2.Design a Python function that accepts a list as input and extracts the three smallest 
numbers from the list, storing each of them in separate variables. After the storage, 
function should return all three variables."""

def smallest(list):
    list.sort()
    return list
list = []
size = int(input("Enter the size of the list:"))
for i in range(size):
    list.append(int(input("Enter the list item:")))
list = smallest(list)
a = list[0]; b=list[1]; c=list[2]
print(a,"\t",b,"\t",c)


"""5.3.Create a Python function that takes a list of names and sorts them alphabetically in 
ascending order."""

def sort_list(list):
    list.sort()
    return list
list = []
size = int(input("Enter the size of the list:"))
for i in range(size):
    list.append(input("Enter the list item:"))
final_list = sort_list(list)
for i in list:
    print(i) 


"""5.4.Write a function that searches for a specific element in a list and returns its index. If 
the element is not found, return -1."""

def search(list,element):
    if element in list:
        return list.index(element)
    else:
        return -1
list = []
size = int(input("Enter the size of the list:"))
for i in range(size):
    list.append(input("Enter the list item:"))
s_element = input("Enter required element:")
s_index = search(list, s_element)
if s_index == -1:
    print(f"Element not found so index is {s_index}")
else:
    print(f"Index of {s_element} is {s_index}")


"""5.5.Create a function that takes list as an input and removes all duplicate elements from a 
list and returns a new list with unique elements."""

def remove_dup(list):
    u_list = []
    for i in list:
        if i not in u_list:
            u_list.append(i)
    return u_list
list = []
size = int(input("Enter the size of the list:"))
for i in range(size):
    list.append(input("Enter the list item:"))
print(remove_dup(list))


"""5.6.Write a function that that takes two lists as an input, concatenates two lists and 
removes any duplicates from the resulting list. And then returns the new list"""

def concate(list1,list2):
    list = list1 + list2
    u_list = []
    for i in list:
        if i not in u_list:
            u_list.append(i)
    return u_list
list1 = []
size = int(input("Enter the size of the list1:"))
for i in range(size):
    list1.append(input("Enter the list item:"))
list2 = []
size = int(input("Enter the size of the list2:"))
for i in range(size):
    list2.append(input("Enter the list item:"))
print(concate(list1,list2))