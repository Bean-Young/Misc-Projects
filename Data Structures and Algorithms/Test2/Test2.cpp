#include <iostream>
using namespace std;
#define OK 1
#define ERROR 0
#define OVERFLOW -2
#define MAXSIZE 100
typedef int Status;
typedef int ElemType;
typedef int QElemType;
typedef struct
{
    ElemType* elem;
    int length;
} SqList;  
typedef struct LNode
{
    ElemType data;
    struct LNode *next;
}LNode, * LinkList;
typedef struct CiLink_QNode
{
    QElemType data;
    struct CiLink_QNode* next;
}CiLink_QNode, * LinkQueue;
Status Init_SqList(SqList& L)
{
    L.elem = new ElemType[MAXSIZE];
    if (!L.elem) exit(OVERFLOW);
    L.length = 0;
    return OK;
}
Status GetElem_SqList(SqList L, int i, ElemType& e)
{
    if (i<1 || i>L.length) return ERROR;
    e = L.elem[i - 1];
    return OK;
}
int LocateElem_SqList(SqList L, ElemType e)
{
    for (int i = 0; i < L.length; i++)
        if (L.elem[i] == e) return i + 1;
    return 0;
}
Status Insert_SqList(SqList& L, int i, ElemType e)
{
    if (i<1 || i>L.length) return ERROR;
    if (L.length == MAXSIZE) return OVERFLOW;
    for (int j = L.length - 1; j >= i - 1; j--)
        L.elem[j + 1] = L.elem[j];
    L.elem[i - 1] = e;
    L.length++;
    return OK;
}
Status Delete_SqList(SqList& L, int i,ElemType &e)
{
    if (i<1 || i>L.length) return ERROR;
    for (int j = i; j <= L.length - 1; j++)
        L.elem[j - 1] = L.elem[j];
    L.length--;
    return OK;
}
Status Init_LinkList(LinkList& L)
{
    L = new LNode;
    if (!L) return ERROR;
    L->next = NULL;
    return OK;
}
Status GetElem_LinkList(LinkList L, int i, ElemType& e)
{
    LNode* p = L->next;
    int j = 1;
    while (p && j << i)
    {
        p = p->next;
        j++;
    }
    if (!p || j > i) return ERROR;
    e = p->data;
    return OK;
}
LNode* LocateElem_LinkList(LinkList L, ElemType e)
{
    LNode* p = L->next;
    while (p && p->data != e)
        p = p->next;
    return p;
}
Status Insert_LinkList(LinkList L, int i, ElemType e)
{
    LNode* p = L;
    int j = 0;
    while (p&&(j<i-1))
    {
        p = p->next; 
        j++;
    }
    if (!p || j > i - 1) return ERROR;
    LNode *s = new LNode;
    s->data = e;
    s->next = p->next;
    p->next = s;
    return OK;
}
Status Delete_LinkList(LinkList L, int i)
{
    LNode* p = L;
    int j = 0;
    while (p && (j < i - 1))
    {
        p = p->next;
        j++;
    }
    if (!p || j > i - 1) return ERROR;
    LNode* q = p->next;
    p->next = q->next;
    delete q;
    return OK;
}
void CreateList_H(LinkList& L,int n)
{
    L = new LNode;
    L->next = NULL;
    for (int i = 0; i < n; i++)
    {
        LNode* p = new LNode;
        cin >> p->data;
        p->next = L->next;
        L->next = p;
    }
}
void CreateList_R(LinkList& L, int n)
{
    L = new LNode;
    L->next = NULL;
    LNode *r= L;
    for (int i = 0; i < n; i++)
    {
        LNode* p = new LNode;
        cin >> p->data;
        p->next = NULL;
        r->next = p;
        r= p;
    }
}
Status Print_LinkList(LinkList& L)
{
    LNode* p = L->next;
    while (p)
    {
        cout << p->data <<"\t";
        p = p->next;
    }
    cout << "\n";
    return OK;
}
Status ReverseInsert(SqList sq, LinkList& L)
{
    L = new LNode;
    L->next = NULL;
    for (int i = 0; i < sq.length; i++)
    {
        LNode* p = new LNode;
        p->data=sq.elem[i];
        p->next = L->next;
        L->next = p;
    }
    return OK;
}
Status Delete(LinkList L, int i, int j)
{
    LNode* p = L; int k = 0;
    while ((p->next) && (k < i - 1))
    {
        p=p->next;
        k++;
    }
    if (!(p->next) || (k > i - 1)) return ERROR;
    for (; k < j; k++)
    {
        LNode* q = p->next;
        p->next = q->next;
        delete q;
        if (!(p->next) && (k != j-1)) return ERROR;
    }
    return OK;
}
Status DeleteDuplicate(LinkList L)
{
    LNode* p = L->next;
    LNode* pre = L;
    while (p->next)
    {
        LNode* q=p->next;
        while (q) if (q->data == p->data) break; else q = q->next;
        if (q)
        {
            LNode* t = p;
            pre->next = p->next;
            p = p->next;
            delete t;
        }
        else
        {
            pre = pre->next;
            p = p->next;
        }
    }
    return OK;
}
LNode* Middle(LinkList L)
{
    LNode* q = L;
    LNode* s = L;
    while (q && q->next)
    {
        q = q->next->next;
        s = s->next;
    }
    return s;
}
Status InitQueue(LinkQueue& Q)
{
    Q = new CiLink_QNode;
    Q->next = Q;
    return OK;
}
Status EnQueue(LinkQueue &Q, QElemType e)
{
    CiLink_QNode * p = new CiLink_QNode;
    if (!p) return(OVERFLOW);
    p->data = e;
    p->next = Q->next;
    Q->next = p;
    Q = Q->next;
    return OK;
}
Status DeQueue(LinkQueue& Q, QElemType& e)
{
    if (Q->next == Q) return ERROR;
    CiLink_QNode* p = Q->next->next;
    e = p->data;
    Q->next->next = p->next;
    if (p == Q) Q = Q->next;
    delete p;
    return OK;
}
int main()
{
    LinkQueue Q;
    InitQueue(Q);
    for (int i = 1;i <= 5;i++)
    {
        int e;
        cin >> e;
        EnQueue(Q, e);
    }
    for (int i = 1; i <= 6;i++)
    {
        int e=0;
        DeQueue(Q, e);
        cout << e;
    }
    return 0;
}