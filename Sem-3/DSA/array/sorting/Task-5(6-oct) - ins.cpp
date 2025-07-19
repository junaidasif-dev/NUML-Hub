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
	//insertion sorting
	for(int i=1; i<size; i++)
	{
		int temp = arr[i];
		int j = i;
		while(j>0 && arr[j-1]>=temp)
		{
			arr[j] = arr[j-1];
			j--;
		}
		arr[j] = temp;
	}
	cout<<"The sorted array is as follows: "<<endl;
	for(int i=0; i<size; i++)
	{
		cout<<arr[i]<<endl;
	}
	return 0;
}
