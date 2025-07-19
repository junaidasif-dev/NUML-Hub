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
	//selection sorting
	for(int i=0; i<size; i++)
	{
		int min = i;
		for(int j=i+1; j<size; j++)
		{
			if(arr[min]>arr[j])
			{
				min = j;
			}
		}
		int temp = arr[i];
		arr[i] = arr[min];
		arr[min] = temp;
	}
	
	cout<<"The sorted array is as follows: "<<endl;
	for(int i=0; i<size; i++)
	{
		cout<<arr[i]<<endl;
	}
	return 0;
}
