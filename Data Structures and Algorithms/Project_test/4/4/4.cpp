#include "stdafx.h"
using namespace std;
#define OK 1
#define ERROR 0
#define OVERFLOW -2
#define MAXLEN 255
typedef char TElemType;
typedef int Status;
typedef struct BiTNode {
	TElemType data;
	struct BiTNode* lchild, * rchild;
}BiTNode,*BiTree;
void CreatBiTree(BiTree& T)
{
	char ch;
	cin >> ch;
	if (ch == '#') T = NULL;
	else
	{
		T = new BiTNode;
		T->data = ch;
		CreatBiTree(T->lchild);
		CreatBiTree(T->rchild);
	}
}
int NodeCount(BiTree T)
{
	if (!T) return 0;
	if ((!T->lchild) && (!T->rchild)) return NodeCount(T->rchild) + NodeCount(T->lchild) + 1;
	return NodeCount(T->rchild) + NodeCount(T->lchild);
}
void FreeBiTree(BiTree T)
{
	if (T)
	{
		FreeBiTree(T->lchild);
		FreeBiTree(T->rchild);
		delete(T);
	}
}
void PostOrderTraverse(BiTree T)
{
	if (T)
	{		
		PostOrderTraverse(T->lchild);
		PostOrderTraverse(T->rchild);
		cout << T->data;
	}
}
int main()
{
	BiTree Tree;
	CreatBiTree(Tree);
	PostOrderTraverse(Tree);
	cout << endl;
	cout << NodeCount(Tree)<<endl;
	FreeBiTree(Tree);
	_CrtDumpMemoryLeaks();
	return 0;
}
