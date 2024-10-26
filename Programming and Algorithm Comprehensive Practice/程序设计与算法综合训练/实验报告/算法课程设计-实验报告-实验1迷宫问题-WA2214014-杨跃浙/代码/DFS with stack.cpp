#include <iostream>
#include <vector>
using namespace std;

#define MAX_SIZE 6

bool can[MAX_SIZE][MAX_SIZE] = { false };

typedef struct Point {
    int x, y;
    vector<int> path;
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
        return { -1, -1, {} };
    }
    Point data = (*head)->data;
    Node* temp = *head;
    *head = (*head)->next;
    delete temp;
    return data;
}

void output(const vector<int>& path) {
    int x = 1, y = 1;
    for (int dir : path) {
        cout << '(' << x << ',' << y <<','<<dir<< ") -> ";
        if (dir == 1) x++;
        else if (dir == 2) x--;
        else if (dir == 3) y++;
        else if (dir == 4) y--;
    }
    cout << '(' << x << ',' << y << ')' << endl; 
}

int main() {
    Node* head = nullptr;
    for (int i = 1; i <= 4; i++) {
        for (int j = 1; j <= 4; j++) {
            int x;
            cin >> x;
            if (x == 0) {
                can[i][j] = true;
            }
        }
    }
    push(&head, { 1, 1, {} }); // 将起点加入栈
    can[1][1] = false;

    while (head) {
        Point current = pop(&head);

        if (current.x == 4 && current.y == 4) { // 到达终点
            output(current.path);
            break; // 找到一条路径就退出
        }

        // 四个方向：1向下，2向上，3向右，4向左
        for (int dir = 1; dir <= 4; ++dir) {
            int nx = current.x, ny = current.y;
            switch (dir) {
            case 1: nx++; break;
            case 2: nx--; break;
            case 3: ny++; break;
            case 4: ny--; break;
            }
            if (can[nx][ny]) {
                can[nx][ny] = false; // 标记为已访问
                Point newPoint = { nx, ny, current.path };
                newPoint.path.push_back(dir); // 添加方向到路径
                push(&head, newPoint); // 加入新点到栈中
            }
        }
    }

    // 清理剩余栈元素
    while (head) pop(&head);

    return 0;
}




































































/*
#include <iostream>
using namespace std;
bool can[6][6] = { false };
typedef struct Point {
    int x, y;
    int path[36];
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
void append(int arr[],int dir,int p1[])
{
    for (int i = 0; i < 36; i++) 
    {
        if (arr[i] == 0) { 
            p1[i] = dir;
            break;
        }
        else
        {
            p1[i] = arr[i];
        }
    }
}
void output(int p1[])
{
    int x =1,y = 1;
    int i = 0;
    while (p1[i]!=0)
    {
        cout << '(' << x << ',' << y << ','<<p1[i] << ')' << endl;
        if (p1[i] == 1) 
        {
            y = y + 1;
        }
        if (p1[i] == 2)
        {
            y = y - 1;
        }
        if (p1[i] == 3)
        {
            x = x + 1;
        }
        if (p1[i] == 4)
        {
            x = x - 1;
        }
    }


}
int main() {
    Node* head = NULL;
    for (int i=1;i<=4;i++)
    {
        for (int j = 1;j <= 4;j++)
        {
            int x;
            cin >> x;
            if (x == 0)
            {
                can[i][j] = true;
            }
        }
    }
    if (can[2][1])
    {
        push(&head, { 2,1 ,{1} });
        can[2][1] = false;
    }
    if (can[1][2])
    {
        push(&head, { 1,2,{3} });
        can[1][2] = false;
    }
    can[1][1] = false;
    while (head) {
        Point p = pop(&head);
        int p1[36] = { 0 };
        if (can[p.x, p.y + 1])
        {
            append(p.path, 1, p1);
            push(&head, { p.x,p.y + 1,*p1 });
            can[p.x][p.y+1] = false;
        }
        if (can[p.x, p.y - 1])
        {
            append(p.path, 2, p1);
            push(&head, { p.x,p.y - 1 ,*p1});
            can[p.x][p.y - 1] = false;
        }
        if (can[p.x + 1, p.y])
        {
            append(p.path, 3, p1);
            push(&head, { p.x + 1,p.y ,*p1});
            can[p.x+1][p.y] = false;
        }
        if (can[p.x - 1, p.y])
        {
            append(p.path, 4, p1);
            push(&head, { p.x - 1,p.y ,*p1});
            can[p.x-1][p.y] = false;
        }
        if ((p.x == 4) && (p.y == 4)) 
        {
            output(p.path);
        }
    }
    getchar();
    return 0;
}
*/