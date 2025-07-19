#include "node.h"
#include <iostream>
using namespace std;

class linkedlist
{
	private:
		Node* head;
		Node* tail;
	public:
		linkedlist()
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
		void deletenode(double value);
		void traverselist();
		bool isempty();
};
bool linkedlist::isempty()
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
void linkedlist::insertathead(double value)
	{
		Node* newnode = new Node(value); 
		if(head == 0 && tail == 0)
		{
			head = tail = newnode;
		}
		else
		{
			newnode->next = head;  
			head = newnode;
		}
	}
void linkedlist::insertattail(double value)
	{
		Node* newnode = new Node(value); 
		if(head == 0 && tail == 0)
		{
			head = tail = newnode;
		}
		else
		{
			tail->next = newnode;
			tail = newnode;
		}
	}
void linkedlist::insertafter(double existing, double value)
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
			Node* currnode = head;
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
				Node* newnode = new Node(value);
				newnode->next = currnode->next;
				currnode->next = newnode;
			}
		}
	}	
void linkedlist::insertbefore(double existing, double value)
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
			Node* prevnode = 0;
			Node* currnode = head;
			while(currnode != 0 && currnode->data != existing)
			{
				prevnode = currnode;
				currnode = currnode->next;
			}
			if(currnode==0)
			{
				cout<<"\nInsertion is not possible in the list because existing element in not present in the list.";
			}
			else
			{
				Node* newnode = new Node(value);
				newnode->next = currnode;    	   // newnode = currnode
				prevnode->next = newnode;          // currnode->next = currnode
			}
		}
	}
void linkedlist::traverselist()	
	{
		if(isempty())
		{
			cout<<"\nList is empty.";
		}
		else
		{
			cout<<"\nValues in list are: "<<endl;
			Node* currnode = head;
			while(currnode != 0)
			{
				cout<<currnode->data<<endl;
				currnode = currnode->next;
			}
		}
	}	
void linkedlist::deletefromhead()
	{
		if(isempty())
		{
			cout<<"\nList is empty.";
		}
		else if(head==tail)
		{
			Node* currnode = head;
			head = tail = 0;
			delete currnode;
		}	
		else
		{
			Node* currnode = head;
			head = head->next;
			currnode->next = 0;
			delete currnode;
		}
	}	
void linkedlist::deletefromtail()
	{
		if(isempty())
		{
			cout<<"\nList is empty.";
		}
		else if(tail==head)
		{
			Node* currnode = tail;
			tail = head = 0;
			delete currnode;
		}	
		else
		{
			Node* currnode = head;
			Node* prevnode = 0;
			while(currnode->next!=0)
			{
				prevnode = currnode;
				currnode = currnode->next;
			}
			tail = prevnode;
			tail->next = 0;
			delete currnode;
		}
	}			
void linkedlist::deletenode(double value)
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
			Node* currnode = head;
			Node* prevnode = 0;
			while(currnode!=0 && currnode->data!=value)
			{
				prevnode = currnode;
				currnode = currnode->next;
			}
			if(currnode==0)
			{
				cout<<"\nValue doesn't exist in the list.";
			}
			else
			{
				prevnode->next = currnode->next;
				currnode->next=0;
				delete currnode;
			}
		}
		
	}		
