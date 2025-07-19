#include "stack.h"
#include <iostream>
using namespace std;

int main() 
{
  int size, num, choice;
  cout << "Enter the Size of Stack: ";
  cin >> size;
  IntStack ist(size);

  do {
    cout << "\n\nPress 1 to PUSH number into the stack.\n";
    cout << "Press 2 to POP number from the stack.\n";
    cout << "Press 3 to DISPLAY the stack.\n";
    cout << "Press 4 to Exit.\n";
    cout << "\n\nEnter Your Choice: ";
    cin >> choice;

    if (choice == 1) 
	{
      cout << "Enter the total numbers you want to push: ";
      cin >> num;
      for (int i = 0; i < num; i++) 
	  {
        ist.push();
      }
    } 
	else if (choice == 2)
	{
      ist.pop();
    } 
	else if (choice == 3) 
	{
      ist.display();
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
