#include<iostream>
using namespace std;
class doubleQueue
{

	private:
		double *queueArray;
		int queuesize;
		int front;
		int rear;
	public:
	doubleQueue(int size)
	{
		queueArray=new double(size);//double[size]
		queuesize=size;
		front=-1;
		rear=-1;
	}
	~doubleQueue()
	{
		delete []queueArray;
		}	
		bool isFull()
		{
			if(rear == queuesize-1)
			{
				return true;
			}
			else
			{
				return false;
			}
		}
		bool isEmpty()
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
		void ENQUEUE(double value)
		{
			if(isFull())
			{
				cout<<"Queue is full :"<<endl;
			}
			else
			{
				if(front==-1)
				{
					front=0;
				}
				rear++;
			cout<<"Insert number:";	
			cin>>queueArray[rear];
			}
		}
		void DEQUEUE()
		{
			if(isEmpty())
			{
				cout<<"Queue is Empty"<<endl;
			}
			else
			{
				cout<<queueArray[front]<<"value is delete from queue"<<endl;
				if(front==rear)//only one item in queue
				{
					front=rear=-1;
				}
				else
				{
					front++;
				}
			}
		}
		void DISPLAY()
		{
			if(isEmpty())
			{
				cout<<"list is empty"<<endl;
			}
			else
			{
				cout<<"values in Queue are"<<endl;
				for(int i=front;i<=rear;i++)
				{
					cout<<queueArray[i]<<"\t"<<endl;
				}
			}
		}
};
