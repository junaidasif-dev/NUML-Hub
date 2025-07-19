#include <iostream>
using namespace std;

class IntStack{

  private:
    int *stackArray;
    int stackSize;
    int top;
    int num;

  public:
    IntStack(int size){
      stackArray = new int[size];
      stackSize = size;
      top = -1;
    }

    ~IntStack(){
      delete [] stackArray;
    }

    void push(){
      
      if(isFull()){
        cout << "\nThe stack is full.\n";
      }
      else{
        top++;

        cout << "\nEnter the Number you want to PUSH: ";
        cin >> num;
        stackArray[top] = num;
        cout << "\nNumber is Pushed into Stack!!\n";
      }
    }



    void pop(){
      
      if(isEmpty()){
        cout << "\nStack Overflow!!\n";
      }
      else{
        cout << "\nThe popped element is: " << stackArray[top] << endl;
        top--;
      }
    }


    bool isFull(){
      
      if(top == stackSize - 1){
        return true;
      }
      else{
        return false;
      }
    }

    bool isEmpty(){
      
      if(top == -1){
        return true;
      }
      else{
        return false;
      }
    }


    void display(){
      
      if(isEmpty()){
        cout << "\nStack Underflow!!\n";
      }
      else{
        cout << "\nThe stack elements are: ";
        
        for(int i = top; i >= 0; i--){  // for(int i = 0; i <= top; i++)
          cout << stackArray[i] << " ";
        }
        cout << endl;
      }
      
    }


};
