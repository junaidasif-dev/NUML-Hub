/*#include<iostream>
#include<math.h>
using namespace std;
char tree[15];
void SetRoot(char val)
{
	if(tree[0]=='\0')
	{
		tree[0]=val;
	}
	else
	{
		cout<<"Root already exist."<<endl;
	}
}
int SearchNode(char val)
{
	int index=-1;
	for(int i=0;i<15;i++)
	{
		if(tree[i]==val)
		{
			index=i;
			break;
		}	
	}
	return index;
}
void SetLeftChild(char prnt, char val)
{
	int index=SearchNode(prnt);
	if(index==-1)
	{
		cout<<"Parent element is not existed in the list."<<endl;
	}
	else
	{
		tree[(2*index)+1]=val;
	}
}
void SetRightChild(char prnt, char val)
{
	int index=SearchNode(prnt);
	if(index==-1)
	{
		cout<<"Parent element is not existed in the list."<<endl;
	}
	else
	{
		tree[(2*index)+2]=val;
	}
}
void PrintTree()
{
	cout<<"Elements in tree are:"<<endl;
	for(int i=0;i<15;i++)
	{
		if(tree[i]!='\0')
		{
			cout<<tree[i];
		}
		else
		{
			cout<<"-";
		}
	}
}
int main()
{
	SetRoot('m');
	SetLeftChild('m', 'o');
	SetRightChild('o', 's');
	PrintTree();
}*/




#include<iostream>
using namespace std;
char tree[15];
void setroot(char val)
{
	if(tree[0]=='\0')
	{
		tree[0]=val;
	}
	else
	{
		cout<<"Root node already exist........."<<endl;
	}
	
}
int searchnode(char val)
{
	int index=-1;
    for (int i=0 ; i<15 ;i++)
    {
     if(tree[i]==val)
      {
    	index=i;
    	break;
	  }
	}
	return index;
}
void setleft(char p,char lc)
{
    int lindex=searchnode(p);
    if(lindex==-1)
    {
    	cout<<"given node is not in list...."<<endl;
	}
	else
	{
		tree[(2*lindex)+1]=lc;
	}
}
void setright(char p,char rc)
{
    int rindex=searchnode(p);
    if(rindex==-1)
    {
    	cout<<"given node is not in list...."<<endl;
	}
	else
	{
		tree[(2*rindex)+2]=rc;
	}
}
void printtree()
{
	cout<<"nodes in tree are as follows......."<<endl;
	for(int i=0 ; i<15 ;i++)
	{
		if(tree[i]!='\0')
		{
	    	cout<<tree[i]<<endl;
    	}
    	else
    	{
    		cout<<"--"<<endl;
		}
	}	
}
int main()
{
	int choise;
	char val,cons;
	char p,lc,rc;
	do
	{
		cout<<"enter your choise..."<<endl;
		cout<<"\tpress 1 for set root..."<<endl;
		cout<<"\tpress 2 for set leftchild..."<<endl;
		cout<<"\tpress 3 for set rightchild..."<<endl;
		cout<<"\tpress 4 for  search node..."<<endl;
		cout<<"\tpress 5 for print tree..."<<endl;
		cin>>choise;
		if(choise==1)
		{
			cout<<"enter the node you want to set root node"<<endl;
			cin>>val;
			setroot(val);	
		}
		if(choise==2)
		{
			cout<<"enter value you want to set on left child"<<endl;
			cin>>p;
			cin>>lc;
			setleft(p,lc);
		}
		if(choise==3)
		{
			cout<<"enter value you want to set on right child"<<endl;
			cin>>p;
			cin>>rc;
			setright(p,rc);
		}
		if(choise==4)
		{
			cout<<"which node you want to search...."<<endl;
			cin>>val;
			searchnode(val);
		}
		if(choise==5)
		{
			printtree();
		}
		if(choise==6)
		{
			cout<<"SORRY!!INVALID CHOISE"<<endl;
		}
		cout<<"press c for continue or press any other key for exit"<<endl;
		cin>>cons;
		
	}
	while(cons=='c');
}
