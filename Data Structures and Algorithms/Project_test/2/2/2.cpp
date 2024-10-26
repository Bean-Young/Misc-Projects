#include "stdafx.h"
using namespace std;
#define OK 1
#define ERROR 0
#define OVERFLOW -2
#define MAXSIZE 100
typedef int Status;
typedef int ElemType;
typedef struct LNode
{
    ElemType data;
    struct LNode* next;
}LNode, * LinkList;
Status InitList(LinkList& L)
{
    L = new LNode;
    if (!L) return ERROR;
    L->next = NULL;
    return OK;
}
Status Overturn(LinkList L)
{
    LinkList p = L->next;
    L->next = NULL;
    while (p)
    {
        LNode* r = p;
        p = p->next;
        r->next = L->next;
        L->next = r;
    }
    return OK;
}
void CreateList(LinkList& L, int n)
{
    L = new LNode;
    L->next = NULL;
    LNode* r = L;
    for (int i = 0; i < n; i++)
    {
        LNode* p = new LNode;
        cin >> p->data;
        p->next = NULL;
        r->next = p;
        r = p;
    }
}
Status PrintList(LinkList& L)
{
    LNode* p = L->next;
    while (p)
    {
        cout << p->data << "\t";
        p = p->next;
    }
    cout << endl;
    return OK;
}
Status DeleteList(LinkList& L)
{
    LNode* p = L;
	while (p->next)
    {
	    LNode* q = p->next;
		p->next = q->next;
		delete q;
    }
    delete p;
    return OK;
}
int main()
{
    LinkList Link;
    InitList(Link);
    CreateList(Link, 5);
    PrintList(Link);
    Overturn(Link);
    PrintList(Link);
    DeleteList(Link);
	_CrtDumpMemoryLeaks();
    return 0;
}
