#include <iostream>
using namespace std;

typedef struct Point {
    int x, y;
} Point;

typedef struct Node {
    Point data;
    struct Node* next;
} Node;

void push(Node** head, Point data) {
    Node* new_node = new Node;
    new_node->data = data;
    new_node->next = *head;
    *head = new_node;
}

Point pop(Node** head) {
    if (*head == NULL) {
        cout << "Stack is empty." << endl;
        return { -1,-1 };
    }
    Point data = (*head)->data;
    Node* temp = *head;
    *head = (*head)->next;
    delete temp;
    return data;
}

int main() {
    Node* head = NULL;
    push(&head, { 1,1 });
    push(&head, { 2,2 });
    push(&head, { 3,3 });

    while (head) {
        Point p = pop(&head);
        cout << "(" << p.x << ", " << p.y << ")" << endl;
    }
    getchar();
    return 0;
}
