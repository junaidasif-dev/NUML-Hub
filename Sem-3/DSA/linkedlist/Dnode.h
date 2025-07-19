#include<iostream>
using namespace std;

class Dnode
{
	public:
		double data;
		Dnode* next;
		Dnode* prev;
		Dnode(double i=0, Dnode* n=0, Dnode* p=0)
		{
			data = i;
			next = n;
			prev = p;
		}
};
