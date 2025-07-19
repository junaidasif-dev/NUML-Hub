#include<iostream>
using namespace std;

class Node
{
	public:
		double data;
		Node* next;
		Node(double i=0, Node* n=0)
		{
			data = i;
			next = n;
		}
};
