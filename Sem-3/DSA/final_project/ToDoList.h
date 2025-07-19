#include<string>
#include"TDnode.h"
#include<iostream>
using namespace std;

class ToDoList 
{
	private:
    	Node* head;
	public:
    	ToDoList()
    	{
    		head = 0;
		}
		bool IsEmpty();
		void AddTask(string Task);
		void DisplayTask();
		void DeleteTask(string Task);
};
bool ToDoList::IsEmpty()
{
	if(head == 0)
	{
		return true;	
	}
	else
	{
		return false;
	}	
}
// Add a new task to the list
void ToDoList::AddTask(string Task) 
{
    Node* newTask = new Node(Task);
    if(IsEmpty()) 
	{
        head = newTask;
    }
	else 
	{
    	Node* currnode = head;
        while (currnode->next != 0) 
		{
            currnode = currnode->next;
        }
        currnode->next = newTask;
        currnode = newTask;
    }
    cout<<"Added task: "<<Task<<endl;
}
// Display all tasks in the list
void ToDoList::DisplayTask() 
{  
    Node* currnode = head;
    while(currnode != 0)
	{
        cout<<"- "<<currnode->data<<endl;
        currnode = currnode->next;
    }
}
// Remove a task from the list
void ToDoList::DeleteTask(string Task) 
{ 
    Node* currnode = head;
    Node* prevnode = 0;
    while(currnode!=0 && currnode->data!=Task) 
	{
        prevnode = currnode;
        currnode = currnode->next;
    }
    if(currnode == 0) 
	{
        cout<<"Task not found."<<endl;
    } 
	else 
	{
        if (prevnode == 0) 
		{
            head = head->next;
            currnode->next = 0;
        } 
		else 
		{
            prevnode->next = currnode->next;
            currnode->next = 0;
        }
        delete currnode;
        cout<<"Task removed: "<<Task<<endl;
    }
}   
