#include<iostream>
using namespace std;
void bubble(int arr[], int size) //bubble sorting
{
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
}
void selection(int arr[], int size) //selection sorting
{
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
}
void insertion(int arr[], int size) //insertion sorting
	{
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
	}
int main()
{
	int size;
  	cout<<"Enter size of array: ";
  	cin>>size;
  	int arr[size];
  	cout<<"Enter values in an array"<<endl;
	for(int i=0; i<size; i++)
	{
		cout<<"Enter valuse: ";
		cin>>arr[i];
	}
	char choice;
	cout<<"Do you want to sort the array or do you want to find a number?\nIf you want to sort the array, then press (s) and if you want to find a number, then press (f): ";
	cin>>choice;
	if(choice == 's')
		{
			char schoice;
			cout<<"Please choose sorting method\nPress (b) for bubble sorting, (s) for selection sorting, or (i) for insertion sorting: ";
			cin>>schoice;
			if(schoice == 'b')
			{
				bubble(arr, size);	
			}
			else if(schoice == 's')
			{
				selection(arr, size);
			}
			else if(schoice == 'i')
			{
				insertion(arr, size);
			}
			else
			{
				cout<<"\nInvalid choice.";
				exit(0);
			}
			cout<<"The sorted array is as follows: "<<endl;
			for(int i=0; i<size; i++)
				{
					cout<<arr[i]<<endl;
				}
		}
	else if(choice == 'f')
		{
			bubble(arr, size);
			int NumtoFind;
    		bool flag = false;
    		int start = 0;
    		int end=size-1;
    		int mid;
    		cout<<"Enter the number you want to find in an array: "<<endl;
    		cin>>NumtoFind;
    		char fchoice;
			cout<<"Please choose searching method \nPress (l) for linear search and press (b) for binary search: "<<endl;
			cin>>fchoice;
			while(start <= end)
				{
					if(fchoice == 'l') //linear search
					{
	          		 if(arr[start] == NumtoFind)
			   			{
	            			flag = true;
	            			break;
	           			}
	           		else
			   			{
	           				start++;
	           			}
	    			}
					else if(fchoice == 'b') //binary search
					{
	        			mid = (start+end)/2;
	        		    if(arr[mid] == NumtoFind)
					   {
	        		    	flag = true;
	            			break;
	           		   }
	            		if(arr[mid]<NumtoFind)
	           			{
	            			start=mid+1;
	           			}
	           			if(arr[mid]>NumtoFind)
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
	        	cout<<NumtoFind<<" is founded in an array at index "<<start<<"\nThis index assigned after sorting the array."<<endl;
	    	}
	    	else
			{
	        	cout<<NumtoFind<<" is not found in an array!";
	    	}
		}
	else
		{
			cout<<"\nInvalid choice.";
		}
}
