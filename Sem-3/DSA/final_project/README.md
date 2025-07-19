                                                        ToDo List Using LinkedList

Description:
	This code represents a To-Do List manager implemented in C++ using a linked list. Let's break down the components:
        Node Class (Node):
            •	Represents a single node in a linked list.
            •	Contains two data members: data (string) to store the task name and next (pointer to Node) to link to the next node.
        To-Do List Class (ToDoList):
            •	Manages the linked list of tasks.
            •	Contains a private member head (pointer to Node) representing the head of the linked list.
        Main Function (main()):
            •	Presents a menu-driven interface to interact with the To-Do List:
                	Adds tasks (Press 1), allowing the user to input task names with spaces using getline(cin, Task).
                	Deletes tasks (Press 2), asking the user for the task name to be deleted.
                	Displays all tasks (Press 3) if the list is not empty.
            •	Uses cin.ignore() after reading integers (choice) to clear the newline character ('\n') from the input buffer before using getline() to read strings. This prevents issues when switching between different input methods.
            •	Continues the program according to the user's choice ('y' for continuing and any other key for exit).

This program provides a basic command-line interface for managing a To-Do List using a linked list, allowing tasks to be added, removed, and displayed. The Node class represents individual elements, while the ToDoList class organizes and manages these elements. The main() function serves as the user interface to interact with the To-Do List functionalities.

Functionality:
    Certainly! Let's break down the functionality of the code:
        •	IsEmpty(): Checks if the list is empty by examining whether the head is pointing to nullptr.
        •	AddTask(string Task): Adds a new task to the end of the list.
        •	DisplayTask(): Displays all tasks in the list.
        •	DeleteTask(string Task): Removes a specific task from the list.
