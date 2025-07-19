#include<iostream>
using namespace std;
class TreeNode
{
	public:
		double info;
		TreeNode* left;
		TreeNode* right;
		TreeNode(double i=0, TreeNode* l=0, TreeNode* r=0)
		{
			info = i;
			left = l;
			right = r;
		}
};
