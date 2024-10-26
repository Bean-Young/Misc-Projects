#include <iostream>
using namespace std;
#define OK 1
#define ERROR 0
#define OVERFLOW -2
#define MAXSIZE 100
#define MAXQSIZE 100
typedef int Status;
typedef int ElemType;
typedef int SElemType;
typedef int QElemType;
typedef struct
{
	SElemType* base;
	SElemType* top;
	int stacksize;
}SqStack;
typedef struct StackNode
{
	ElemType data;
	struct StackNode* next;
}StackNode,*LinkStack;
typedef struct QNode
{
	QElemType data;
	struct QNode* next;
}QNode,*QueuePtr;
typedef struct
{
	QueuePtr front;
	QueuePtr rear;
}LinkQueue;
Status Init_SqStack(SqStack &S)
{
	S.base = new SElemType[MAXSIZE];
	if (!S.base) return(OVERFLOW);
	S.top = S.base;
	S.stacksize = MAXSIZE;
	return OK;
}
Status Push_SqStack(SqStack *S, SElemType e)
{
	if (S->top - S->base == MAXSIZE) return ERROR;
	*(S->top++) = e;
	return OK;
}
Status Pop_SqStack(SqStack* S,SElemType &e)
{
	if (S->top == S->base) return ERROR;
	e = *--S->top;
	return OK;
}
Status GetTop_SqStack(SqStack* S)
{
	if (S->top == S->base) return ERROR;
	return *(S->top - 1);
}
Status Init_LinkStack(LinkStack& S)
{
	S = NULL;
	return OK;
}
Status Push_LinkStack(LinkStack& S, SElemType e)
{
	StackNode* p = new StackNode;
	p->data = e;
	p->next = S;
	S = p;
	return OK;
}
Status Pop_LinkStack(LinkStack & S, SElemType& e)
{
	if (S==NULL) return ERROR;
	e = S->data;
	StackNode* p = S;
	S = S->next;
	delete p;
	return OK;
}
SElemType GetTop_LinkStack(LinkStack S)
{
	if (S != NULL) return S->data;
}
void solve_SqStack(SqStack& S)
{
	for (int i = 1; i <= 5; i++)
	{
		int n;
		cin >> n;
		Push_SqStack(&S, n);
	}
}
void solve_Sq_to_Link(SqStack& S, LinkStack& L)
{
	int e;
	while (Pop_SqStack(&S,e)) Push_LinkStack(L, e);
}
void solve_LinkStack(LinkStack& L)
{
	int e;
	while (Pop_LinkStack(L, e)) cout << e << "\t";
	cout << endl;
}
typedef struct
{
	QElemType* base;
	int front;
	int rear;
}SqQueue;
Status Init_SqQueue(SqQueue& Q)
{
	Q.base = new QElemType[MAXQSIZE];
	if (!Q.base) return(OVERFLOW);
	Q.front = Q.rear = 0;
	return OK;
}
Status En_SqQueue(SqQueue& Q, QElemType e)
{
	if ((Q.rear + 1) % MAXQSIZE == Q.front) return ERROR;
	Q.base[Q.rear] = e;
	Q.rear = (Q.rear + 1) % MAXQSIZE;
	return OK;
}
Status De_SqQueue(SqQueue& Q, QElemType& e)
{
	if (Q.front == Q.rear) return ERROR;
	e = Q.base[Q.front];
	Q.front = (Q.front + 1) % MAXQSIZE;
	return OK;
}
Status Init_LinkQueue(LinkQueue &Q)
{
	Q.front = Q.rear = new QNode;
	Q.front->next = NULL;
	return OK;
}
Status En_LinkQueue(LinkQueue& Q, QElemType e)
{
	QNode* p = new QNode;
	p->data = e;
	p->next = NULL;
	Q.rear->next = p;
	Q.rear = p;
	return OK;
}
Status De_LinkQueue(LinkQueue& Q, QElemType& e)
{
	if (Q.front == Q.rear) return ERROR;
	QNode* p = Q.front->next;
	e = p->data;
	Q.front->next = p->next;
	if (Q.front->next==NULL) Q.rear = Q.front;
	delete p;
	return OK;
}
void solve_SqQueue(SqQueue &Q)
{
	int n;
	for (int i = 1; i <= 5; i++)
	{
		cin >> n;
		En_SqQueue(Q, n);
	}
}
void solve_Sq_to_Link_Q(SqQueue& Q, LinkQueue& L)
{
	int e;
	for (int i = 1; i <= 2; i++)
	{
		De_SqQueue(Q, e);
		En_LinkQueue(L, e);
	}
}
void solve_LinkQueue(LinkQueue& L)
{
	int e;
	for (int i = 1; i <= 2; i++)
	{
		De_LinkQueue(L, e);
		cout << e << "\t";
	}
	cout << endl;
}
bool Matching()
{
	LinkStack S;
	Init_LinkStack(S);
	int flag = 1;
	int flag_c;
	char ch;
	cin >> ch;
	while (ch != '#' && flag)
	{
		if (ch == '[')
		{
			flag_c = 1;
			Push_LinkStack(S, flag_c);
		}
		if (ch == ']')
		{
			if (Pop_LinkStack(S, flag_c)) void;  else flag = 0;
		}
		cin >> ch;
	}
	if ((S == NULL) && flag) return true; else return false;
}
int main()
{
	SqStack Sq;
	LinkStack Link;
	SqQueue Q;
	LinkQueue LinkQ;
	Init_SqStack(Sq);
	Init_LinkStack(Link);
	solve_SqStack(Sq);
	solve_Sq_to_Link(Sq, Link);
	cout<<GetTop_LinkStack(Link)<<endl;
	solve_LinkStack(Link);
	Init_SqQueue(Q);
	Init_LinkQueue(LinkQ);
	solve_SqQueue(Q);
	solve_Sq_to_Link_Q(Q, LinkQ);
	solve_LinkQueue(LinkQ);
	if (Matching()) cout << "Yes"; else cout << "No";
	return 0;
}