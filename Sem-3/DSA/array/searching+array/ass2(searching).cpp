#include<iostream>
using namespace std;
int main()
{
    int size;
    int NumtoFind;
    bool flag = false;
    int mid;
    cout<<"Enter the size of an array: ";
    cin>>size;
    int start = 0;
    int end=size-1;
    int array[size];
    cout<<"Enter the values in an array (in ascending order)"<<endl;
    for(int i=0;i<size;i++)
    {
        cout<<"Enter value "<<i+1<<": ";
        cin>>array[i];
    }
    cout<<"Enter the number you want to find in array: "<<endl;
    cin>>NumtoFind;
    int choice;
	cout<<"Please choose searching method \nPress 1 for linear search and press 2 for binary search: "<<endl;
	cin>>choice;
	while(start <= end)
	{
	if(choice == 1)
		{	//linear search
	        if(array[start] == NumtoFind)
			{
	        	flag = true;
	            break;
	        }
	        else
			{
	            start++;
	        }
	    }
	else if(choice == 2)
		{   //binary search
	        mid = (start+end)/2;
	           if(array[mid] == NumtoFind)
			   {
	            flag = true;
	            break;
	           }
	           if (array[mid]<NumtoFind)
	           {
	             start=mid+1;
	           }
	           if (array[mid]>NumtoFind)
	           {
	             end=mid-1;
	           }          
		}
	else
		{
			cout<<"Invalid choice.";
			exit(0);
		}
	}
	    if(flag)
	    {
	        cout<<NumtoFind<<" is founded in an array at index "<<start<<endl;
	    }
	    else{
	        cout<<NumtoFind<<" is not found in an array!";
	    }
} 
