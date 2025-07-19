#include<iostream>
using namespace std;
class doublecqueue{
	private:
		double *queueArray;
		int queuesize;
		int front;
		int rear;
	public:
		doublecqueue(int size)
		{
			queueArray=new double[size];//double[size];
			queuesize=size;
			front=rear=-1;
		}
		~doublecqueue()
		{
			delete [] queueArray;
		}
	bool isfull()
	{
		if((front==0 && rear==queuesize-1 )|| rear==front-1)
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	bool isempty()
	{
		if(front==-1 && rear==-1)
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	void Enqueue(double val)
	{
		if(isfull())
		{
			cout<<"Queue is full\n";
		}
		else
		{
			if(front==-1)
			{
				front=0;
			}
			rear=(rear+1)%queuesize;
			cout<<"Insert number:";
			cin>>queueArray[rear];
		}
	}
	void Dequeue()
	{
		if(isempty())
		{
			cout<<"Queue is empty\n";
		}
		else
		{
			cout<<queueArray[front]<<" value is Dequeued from Queue .\n";
			if(front==rear)//only one item in the queue
			{
				front=rear=-1;
			}
			else
			{
				front=(front+1)%queuesize;
			}
		}
	}
	void Display()
	{
		if(isempty())
		{
			cout<<"Queue is empty\n";
		}
		else
		{
			if(front<=rear)
			{
				cout<<"Values in queue are :\n";
			    for( int i=front;i<=rear;i++)
			    {
				   cout<<queueArray[i]<<"\t\n";
			    }
			}
			else if (front>rear)
			{
				cout<<"Values in queue are :\n";
				for(int i=front;i<queuesize;i++)
				{
					cout<<queueArray[i]<<"\t\n";
				}
				for (int i=0;i<=rear;i++)
				{
					cout<<queueArray[i]<<"\t\n";
				}
			}
			
		}
	}
};
