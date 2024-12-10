#include "stdafx.h"
using namespace std;
#define OK 1
#define ERROR 0
#define OVERFLOW -2
#define MAXSIZE 100
typedef int Status;
typedef int ElemType;
typedef struct
{
    ElemType* elem;
    int length;
} SqList;
Status InitList(SqList& L)
{
    L.elem = new ElemType[MAXSIZE];
    if (!L.elem) exit(OVERFLOW);
    L.length = 0;
    return OK;
}
Status CreateList(SqList &L,int n)
{
    for (int i=1; i <= n; i++)
    {
        cin >> L.elem[i];
        L.length++;
    }
    return OK;
}
Status PrintList(SqList L)
{
    for (int i = 1; i <= L.length; i++)
        cout << L.elem[i]<<"\t";
    cout << endl;
    return OK;
}
int Partition(SqList &L,int low,int high)
{
	L.elem[0]=L.elem[low];
	int pivotkey=L.elem[low];
	while (low<high)
	{
		while (low<high&&L.elem[high]>=pivotkey) --high;
		L.elem[low]=L.elem[high];
		while (low<high&&L.elem[low]<=pivotkey) ++low;
		L.elem[high]=L.elem[low];
	}
	L.elem[low]=L.elem[0];
	return low;
}
void QSort(SqList &L,int low,int high)
{
	if (low<high)
	{
		int pivotloc=Partition(L,low,high);
		QSort(L,low,pivotloc-1);
		QSort(L,pivotloc+1,high);
	}
}
int main()
{
    int n,num;
    SqList Sq;
    InitList(Sq);
    CreateList(Sq,5);
	QSort(Sq,1,Sq.length);
	PrintList(Sq);
    return 0;
}