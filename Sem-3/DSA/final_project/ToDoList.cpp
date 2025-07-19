#include<string>
#include"ToDoList.h"
#include<iostream>
using namespace std;
int main()
{
	string Task;
	char con;
	int choice;
	ToDoList td;
	do
	{
		cout<<"\tPress 1 for adding the task in the list"<<endl;
		cout<<"\tPress 2 for deleting the task from the list"<<endl;
		cout<<"\tPress 3 for display the tasks from the list"<<endl;
		cin>>choice;
		cin.ignore();
		switch(choice)
		{
		case 1:
			cout<<"Enter the task that you want to add in the list: ";
			getline(cin, Task);
			td.AddTask(Task);
			break;
	     case 2:
    		if(td.IsEmpty()) 
			{
        		cout<<"List is empty."<<endl;
    		}
    		else
			{
				cout<<"Enter the task that you want to remove from the list: ";
				getline(cin, Task);
				td.DeleteTask(Task);
			}
		    break;
		case 3:
			if(td.IsEmpty()) 
			{
        		cout<<"List is empty."<<endl;
    		}
    		else
    		{
				cout<<"Tasks in the list are:"<<endl;
				td.DisplayTask();
			}
		    break;
		default:
			cout<<"Sorry! Wrong choice"<<endl;
			break;		    
		}
		cout<<"\nPress (y) for again continue the program and press any key except (y) for exit: ";
		cin>>con;
	}
	while(con == 'y');
	return 0;
}
