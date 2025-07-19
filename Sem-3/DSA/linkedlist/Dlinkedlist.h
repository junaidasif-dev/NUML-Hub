#include "Dnode.h"
#include <iostream>
using namespace std;

class Dlinkedlist
{
	private:
		Dnode* head;
		Dnode* tail;
	public:
		Dlinkedlist()
		{
			head = 0;
			tail = 0;
		}	
		void insertathead(double value);
		void insertattail(double value);
		void insertafter(double existing, double value);
		void insertbefore(double existing, double value);
		void deletefromhead();
		void deletefromtail();
		void deleteSnode(double value);
		void traverselist();
		void reversetraverselist();
		bool isempty();
};
bool Dlinkedlist::isempty()
	{
		if(head == 0 && tail == 0)
		{
			return true;	
		}
		else
		{
			return false;
		}	
	}
void Dlinkedlist::insertathead(double value)
	{
		Dnode* newnode = new Dnode(value); 
		if(isempty())
		{
			head = tail = newnode;
		}
		else
		{
			newnode->next = head;  
			head->prev = newnode;
			head = newnode;
		}
	}
void Dlinkedlist::insertattail(double value)
	{
		Dnode* newnode = new Dnode(value); 
		if(isempty())
		{
			head = tail = newnode;
		}
		else
		{
			newnode->prev = tail;
			tail->next = newnode;
			tail = newnode;
		}
	}
void Dlinkedlist::insertafter(double existing, double value)
	{
		if(isempty())
		{
			cout<<"\nList is empty.";
		}
		else if(existing == tail->data)
		{
			insertattail(value);
		}
		else
		{
			Dnode* currnode = head;
			while(currnode != 0 && currnode->data != existing)
			{
				currnode = currnode->next;
			}
			if(currnode==0)
			{
				cout<<"\nInsertion is not possible in the list because existing element in not present in the list.";
			}
			else
			{
				Dnode* newnode = new Dnode(value);
				newnode->next = currnode->next;
				newnode->prev = currnode;
				currnode->next->prev = newnode;
				currnode->next = newnode;
			}
		}
	}	
void Dlinkedlist::insertbefore(double existing, double value)
	{
		if(isempty())
		{
			cout<<"\nList is empty.";
		}
		else if(existing == head->data)
		{
			insertathead(value);
		}
		else
		{
			Dnode* currnode = head;
			while(currnode != 0 && currnode->data != existing)
			{
				currnode = currnode->next;
			}
			if(currnode==0)
			{
				cout<<"\nInsertion is not possible in the list because existing element in not present in the list.";
			}
			else
			{
				Dnode* newnode = new Dnode(value);
				newnode->next = currnode;  
				newnode->prev = currnode->prev;
				currnode->prev->next = newnode;
				currnode->prev = newnode;  	   
			}
		}
	}
void Dlinkedlist::traverselist()	
	{
		if(isempty())
		{
			cout<<"\nList is empty.";
		}
		else
		{
			cout<<"\nValues in list are: "<<endl;
			Dnode* currnode = head;
			while(currnode != 0)
			{
				cout<<currnode->data<<endl;
				currnode = currnode->next;
			}
		}
	}	
void Dlinkedlist::reversetraverselist()	
	{
		if(isempty())
		{
			cout<<"\nList is empty.";
		}
		else
		{
			cout<<"\nValues in list are: "<<endl;
			Dnode* currnode = tail;
			while(currnode != 0)
			{
				cout<<currnode->data<<endl;
				currnode = currnode->prev;
			}
		}
	}	
void Dlinkedlist::deletefromhead()
	{
		if(isempty())
		{
			cout<<"\nList is empty.";
		}
		else if(head==tail)
		{
			Dnode* dellnode = head;
			head = tail = 0;
			delete dellnode;
		}	
		else
		{
			Dnode* dellnode = head;
			head->next->prev = 0;
			head = head->next;
			dellnode->next = 0;
			delete dellnode;
		}
	}	
void Dlinkedlist::deletefromtail()
	{
		if(isempty())
		{
			cout<<"\nList is empty.";
		}
		else if(tail==head)
		{
			Dnode* dellnode = tail;
			tail = head = 0;
			delete dellnode;
		}	
		else
		{
			Dnode* dellnode = tail;
			tail->prev->next = 0;
			tail = tail->prev;
			dellnode->prev = 0;
			delete dellnode;
		}
	}			
void Dlinkedlist::deleteSnode(double value)
	{
		if(isempty())
		{
			cout<<"\nList is empty.";
		}
		else if(head->data == value)
		{
			deletefromhead();
		}
		else if(tail->data == value)
		{
			deletefromtail();
		}
		else
		{
			Dnode* dellnode = head;
			while(dellnode!=0 && dellnode->data!=value)
			{
				dellnode = dellnode->next;
			}
			if(dellnode==0)
			{
				cout<<"\nValue doesn't exist in the list.";
			}
			else
			{
				dellnode->prev->next = dellnode->next;
				dellnode->next->prev = dellnode->prev;
				dellnode->next = 0;
				dellnode->prev = 0;
				delete dellnode;
			}
		}
		
	}		
