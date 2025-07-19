#include<iostream>
using namespace std;
int main()
{
	int arr[12];
	cout<<"Enter 12 random integers:"<<endl;
	for(int i=0; i<12; i++)
	{
		cout<<"Enter "<<i+1<<" integer:";
		cin>>arr[i];
	}
	for(int i=0; i<12; i++)
	{
		if(arr[i]%2==0)
		{
			cout<<arr[i]<<endl;
		}
		else
		{
			continue;
		}
	}
	for(int i=0; i<12; i++)
	{
		if(arr[i]%2!=0)
		{
			cout<<arr[i]<<endl;
		}
		else
		{
			continue;
		}
	}
	return 0;
}
