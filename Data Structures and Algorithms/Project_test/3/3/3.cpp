#include "stdafx.h"
using namespace std;
#define OK 1
#define ERROR 0
#define OVERFLOW -2
#define MAXSIZE 100
typedef int Status;
typedef char SElemType;
typedef struct
{
	SElemType* base;
	SElemType* top;
	int stacksize;
}SqStack;
Status InitStack(SqStack& S)
{
	S.base = new SElemType[MAXSIZE];
	if (!S.base) return(OVERFLOW);
	S.top = S.base;
	S.stacksize = MAXSIZE;
	return OK;
}
Status Push(SqStack* S, SElemType e)
{
	if (S->top - S->base == MAXSIZE) return ERROR;
	*(S->top++) = e;
	return OK;
}
Status Pop(SqStack* S, SElemType& e)
{
	if (S->top == S->base) return ERROR;
	e = *--S->top;
	return OK;
}
int main()
{
	SqStack S;
	InitStack(S);
	int flag = 1;
	int flag_c;
	char ch;
	cin >> ch;
	while (ch != '#' && flag)
	{
		if (ch == '[')
			Push(&S, ch);
		if (ch == ']')
			if (Pop(&S, ch)) void;  else flag = 0;
		cin >> ch;
	}
	if ((S.top==S.base) && flag) cout<<"Yes"<<endl; else cout<<"No"<<endl;
	return 0;
}