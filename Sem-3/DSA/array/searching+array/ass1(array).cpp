#include<iostream>
#include<conio.h>
using namespace std;
int main()
{
	int row,column;
	cout<<"Enter number of rows:"<<endl;
	cin>>row;
	cout<<"Enter number of columns:"<<endl;
	cin>>column;
	int a[row][column];
	for(int i=0;i<row;i++)
	{
		for(int j=0;j<column;j++){
		cout<<"Enter row "<<i+1<<" & column "<<j+1<<endl;
		cin>>a[i][j];
		}
		
	}
	int sumofrow[row];
	for(int i=0;i<row;i++)
	{
		int sum=0;
		for(int j=0;j<column;j++){	
		sum=sum+a[i][j];
		sumofrow[row]=sum;
		}
		cout<<"Sum of row "<<i+1<<" is: "<<sumofrow[row]<<endl;
	}
	
	
	return 0;
}
