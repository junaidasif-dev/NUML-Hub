#include "queue.h"
#include <iostream>
using namespace std;

int main() 
{
  int size, num, choice, value;
  cout << "Enter the Size of Queue: ";
  cin >> size;
  doubleQueue dq(size);

  do {
    cout << "\n\nPress 1 to insert the number into the Queue.\n";
    cout << "Press 2 to delete the number from the Queue.\n";
    cout << "Press 3 to DISPLAY the Queue.\n";
    cout << "Press 4 to Exit.\n";
    cout << "\n\nEnter Your Choice: ";
    cin >> choice;

    if (choice == 1) 
	{
      cout << "Enter the total numbers you want to inert: ";
      cin >> num;
      for (int i = 0; i < num; i++) 
	  {
        dq.ENQUEUE(value);
      }
    } 
	else if (choice == 2)
	{
      dq.DEQUEUE();
    } 
	else if (choice == 3) 
	{
      dq.DISPLAY();
    } 
	else if (choice == 4) 
	{
      break;
    } 
	else 
	{
      cout << "Invalid Choice...!!";
    }
  } 
  while (true);
}
