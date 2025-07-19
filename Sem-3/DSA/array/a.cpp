#include<iostream>
using namespace std;
int main()
{
	int a[15];
	for(int i=0;i<15;i++)
	{
		cout<<"\nEnter "<<i<<" number:";
		cin>>a[i];
	}
	int min = a[0];
	for(int i=1;i<15;i++)
	{
		if(min>a[i])
		{
			min = a[i];
		}
	}
	cout<<min;
}
