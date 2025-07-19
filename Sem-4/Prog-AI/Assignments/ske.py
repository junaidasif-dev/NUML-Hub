record = [('Junaid', 13, 21, 13, 17, 20), ('Ali', 14, 19, 13, 17, 18), ('Kashif', 13, 21, 10, 5, 7)]
l = [1, 2, 3]
stud = {record[i][0]: l[i] for i in range(len(record))}  # Construct the dictionary using a dictionary comprehension
print(stud)
ft = filter(lambda item: item[1] < 2, stud.items())  # Filter out items where the value is less than 2
f = dict(ft)  # Convert the filtered items back to a dictionary
print(f)
