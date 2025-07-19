#include"dfnode.h"
#include<iostream>
using namespace std;
class TreeLinkedList
{
	private:
		TreeNode* root;
		TreeNode* loc;   //location
		TreeNode* ploc;  //parent location
		TreeNode* sloc;  //successor location
		TreeNode* psloc; //parent of successor
	public:
		TreeLinkedList()
		{
			root = 0;
		}
		void InOrderTraversal(TreeNode* node);
		void PreOrderTraversal(TreeNode* node);
		void PostOrderTraversal(TreeNode* node);
		void InsertNode(double element);
		void FindSuccessor(TreeNode* node);
		void DeleteNode(double exist);
		bool SearchNode(double exist);
		TreeNode* getRoot();	
};
void TreeLinkedList::InsertNode(double element)
{
	TreeNode* newNode = new TreeNode(element);
	if(root == 0)
	{
		root = newNode;
	}
	else
	{
		TreeNode* parent;
		TreeNode* curr=root;
		while(curr!=0)
		{
			parent = curr;
			if(element < curr->info)
			{
				curr = curr->left;
			}
			else
			{
				curr = curr->right;
			}
		}
		if(parent->info < element)
		{
			parent->right = newNode;
		}
		else
		{
			parent->left = newNode;
		}
	}
}
void TreeLinkedList::PreOrderTraversal(TreeNode* node)
{
	if(node!=0)
	{
		cout<<node->info<<"\t";
		PreOrderTraversal(node->left);  //Recursion
		PreOrderTraversal(node->right);
	}
}
