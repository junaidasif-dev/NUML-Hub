#include<iostream>
#include"Clinkedlist.h"
using namespace std;
int main()
{
	clinkedlist c;
	cout<<"Enter the value you want to insert at head\n";
	double h;
	cin>>h;
	c.insertathead(h);
	cout<<"Enter the value you want to insert at last\n";
	double l;
	cin>>l;
	c.insertatlast(l);
	cout<<"Enter the value you want to insert after\n";
	double s;
	cin>>s;
	cout<<"Enter the existing value after you want to insert\n";
	double existing;
	cin>>existing;
	c.insertafter(existing,s);
	cout<<"Enter the value you want to insert before\n";
	double b;
	cin>>b;
	cout<<"Enter the existing value before you want to insert before \n";
	double existing1;
	cin>>existing1;
	c.insertbefore(existing1,b);
	c.traverse();
	c.deletehead();
	cout<<"THe List after head deletion\n";
	c.traverse();
	cout<<"THe List after last deletion\n";
	c.deletespecific(l);
	c.traverse();
}
