#include<iostream>
using namespace std;

class cnode
{
	public:
		double data;
		cnode* next;
		cnode(double i=0, cnode* n=0)
		{
			data = i;
			next = n;
		}
};
