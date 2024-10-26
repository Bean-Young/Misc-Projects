#include <iostream>
using namespace std;

#define MAX_SIZE 5

// 定义Point结构体 包含坐标x，y
typedef struct Point {
    int x, y;
} Point;

typedef struct Stack {
    Point data[MAX_SIZE];
    int top;
} Stack;

// 向栈中压入一个Point
void push(Stack* s, Point p) {
    if (s->top == MAX_SIZE - 1) {
        cout << "错误：上溢出" << endl;
        return;
    }
    s->data[++s->top] = p;
}

// 从栈中弹出一个Point
Point pop(Stack* s) {
    Point x;
    if (s->top == -1) {
        cout << "错误：下溢出" << endl;
        return { -1, -1 };
    }
    x = s->data[s->top];
    s->data[s->top] = { 0,0 };
    s->top--;
    return x;
}

int main() {
    Stack s = { {{0, 0}}, -1 }; // 初始化栈

    // 向栈中压入点
    push(&s, { 1, 2 });
    push(&s, { 3, 4 });
    push(&s, { 5, 6 });
    push(&s, { 7, 8 });
    push(&s, { 9, 10 });


    while (s.top >= 0) {
        Point p = pop(&s);
        cout << "(" << p.x << ", " << p.y << ")" << endl;
    }

    //system("pause");
    getchar();
    return 0;
}
