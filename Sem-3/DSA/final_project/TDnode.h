#include<string>
#include<iostream>
using namespace std;
class Node
{
	public:
    	string data;
    	Node* next;
		Node(string d=0, Node* n=0)
		{
			data = d;
			next = n;
		}
};
