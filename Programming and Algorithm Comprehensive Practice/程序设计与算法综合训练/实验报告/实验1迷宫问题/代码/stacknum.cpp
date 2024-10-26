#include <iostream>

using namespace std;

#define MAX_SIZE 5

typedef struct Stack {
    int data[MAX_SIZE];
    int top;
} Stack;

void push(Stack* s, int x) {
    if (s->top == MAX_SIZE - 1) {
        cout << "错误：上溢出" << endl;
        return;
    }
    s->data[++s->top] = x;
}

int pop(Stack* s) {
    int x;
    if (s->top == -1) {
        cout << "错误：下溢出" << endl;
        return -1;
    }
    x = s->data[s->top];
    s->data[s->top] = 0;
    s->top--;
    return x;
}

int main() {
    Stack s = { {0}, -1 }; // 初始化
    push(&s, 1);
    push(&s, 2);
    push(&s, 3);
    push(&s, 4);
    push(&s, 5);
    while (s.top >= 0)
        cout << pop(&s) << std::endl;
    pop(&s);
    system("pause");
    // getchar();
    return 0;
}
