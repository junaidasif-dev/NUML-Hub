#include<iostream>
using namespace std;
int main()
{
	int size;
	cout<<"Enter size of array: ";
	cin>>size;
	int arr[size];
	for(int i=0; i<size; i++)
	{
		cout<<"Enter valuse: ";
		cin>>arr[i];
	}
	//bubble sorting
	for(int i = 0; i<size-1; i++)
	{
		for(int j=0; j<size-i-1; j++)
		{
			if(arr[j]>arr[j+1])
				{
					int temp = arr[j];
					arr[j] = arr[j+1];
					arr[j+1] = temp;
					
				}
		}
	}
	cout<<"The sorted array is as follows: "<<endl;
	for(int i=0; i<size; i++)
	{
		cout<<arr[i]<<endl;
	}
	return 0;
}
