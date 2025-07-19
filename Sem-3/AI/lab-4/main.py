"""for x in "apple":
    print(x)"""

"""x = "Apple"
print(len(x))"""

"""txt = "The best things in life are free!"
print("ali" in txt)"""

"""txt = "The best things in life are free!"
if "free" in txt:
    print("Yes, 'free' is present.")
else:
    print("not present")"""

"""txt = "The best things in life are free!"
print("expensive" not in txt)"""

"""txt = "The best things in life are free!"
if "expensive" not in txt:
    print("No, 'expensive' is NOT present.")"""

"""str = "Javatpoint"
print(str[0:])
print(str[1:5])
print(str[2:4])
print(str[:3])
print(str[4:7])
print(str[::-1])"""

"""a = "Hello, World!"
print(a.upper())
print(a.replace("Hello", "Junaid"))"""

"""thislist = ["apple", "banana", "cherry", 11]
print(thislist)
print(len(thislist))"""

"""a = [1, 2, 3] + [4, 5,6]
print(a)

b = ['Hi!'] * 4
print(b)

c = 2 in [1, 2, 3]
print(c)"""

"""cats = ['Tom', 'Snappy', 'Kitty', 'Jessie', 'Chester']
print(cats[2])
#add 
cats.append("Oscar")
print(len(cats))
#Remove 2nd cat, Snappy.
del cats[1]
print(cats)"""

#compare
string1 = "Hello"
string2 = "Junaid"
if string1 == string2:
  print("The two strings are equal.")
else:
  print("The two strings are not equal.")

#join
a = "Hello"
b = " World"
print(a+b)

#Python program to find number of vowels in a given string
vowels = ['a', 'e', 'i', 'o', 'u']
str = "Junaid"
v = 0
for i in str:
    if str in ['a', 'e', 'i', 'o', 'u']:
        v = v+1
print(v)