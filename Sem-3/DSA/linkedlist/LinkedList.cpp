/*#include <iostream>
#include "linkedlist.h"
using namespace std;
int main()
{
	linkedlist list;
	list.insertattail(14);
	list.insertathead(5);
	list.insertathead(26);
	list.insertafter(26,9);
	list.insertbefore(5,12);
	list.traverselist();
	return 0;
}*/

#include <iostream>
#include "linkedlist.h"
using namespace std;
int main()
{
	double val;
	double existing;
	char con;
	int choice;
	linkedlist list;
	do
	{	
		cout<<"\tPress 1 for insert at head"<<endl;
		cout<<"\tPress 2 for insert at tail"<<endl;
		cout<<"\tPress 3 for insert after"<<endl;
		cout<<"\tPress 4 for insert before"<<endl;
		cout<<"\tPress 5 for delete from head"<<endl;
		cout<<"\tPress 6 for delete from tail"<<endl;
		cout<<"\tPress 7 for delete from specific node"<<endl;
		cout<<"\tPress 8 for traverse node"<<endl;
		cout<<"Enter choice: ";
		cin>>choice;
		switch (choice)
		{
			case 1:
			cout<<"Enter value to insert at head: ";
			cin>>val;
			list.insertathead(val);
			break;
	     case 2:
			cout<<"Enter value to insert at tail: ";
			cin>>val;
			list.insertattail(val);
		    break;
		case 3:
			cout<<"Enter value to insert after: ";
			cin>>existing;
			cin>>val;
			list.insertafter(existing,val);
		     break;
	    case 4:
			cout<<"Enter value to insert before: ";
			cin>>existing;
			cin>>val;
			list.insertbefore(existing,val);
		    break;
		case 5:		
			list.deletefromhead();
		    break;
		case 6:		
			list.deletefromtail();
			break;
		case 7:	
			cout<<"Enter value for specific node deletion: ";
			cin>>val;
			list.deletenode(val);
			break;	
		case 8:	
			list.traverselist();
		    break;    
		default:
			cout<<"Sorry! Wrong choice"<<endl;
			break;
		}
		cout<<"\nPress (y) for again continue the program and press any key except (y) for exit: ";
		cin>>con;
		
	}
	while(con == 'y');
}
