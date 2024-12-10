#include <iostream>
#include <string>
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
typedef BiTree QElemType;
typedef struct QNode
{
	QElemType data;
	struct QNode* next;
}QNode, * QueuePtr;
typedef struct
{
	QueuePtr front;
	QueuePtr rear;
}LinkQueue;
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
void PreOrderTraverse(BiTree T)
{
	if (T)
	{
		cout << T->data;
		PreOrderTraverse(T->lchild);
		PreOrderTraverse(T->rchild);
	}
}
void InOrderTraverse(BiTree T)
{
	if (T)
	{
		InOrderTraverse(T->lchild);
		cout << T->data;
		InOrderTraverse(T->rchild);
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
void FreeBiTree(BiTree T)
{
	if (T)
	{
		FreeBiTree(T->lchild);
		FreeBiTree(T->rchild);
		delete(T);
	}
}
int NodeCount(BiTree T)
{
	if (!T) return 0;
	return NodeCount(T->rchild) + NodeCount(T->lchild) + 1;
}
int NodeCount_Leaf(BiTree T)
{
	if (!T) return 0;
	if ((!T->lchild) && (!T->rchild)) return NodeCount_Leaf(T->rchild) + NodeCount_Leaf(T->lchild) + 1;
	return NodeCount_Leaf(T->rchild) + NodeCount_Leaf(T->lchild);
}

Status Init_LinkQueue(LinkQueue& Q)
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
	if (Q.front->next == NULL) Q.rear = Q.front;
	delete p;
	return OK;
}
void LevelOrderTraverse(BiTree T)
{
	LinkQueue Tree;
	BiTree e;
	Init_LinkQueue(Tree);
	En_LinkQueue(Tree,T);
	while (De_LinkQueue(Tree, e))
	{
		cout << e->data;
		if (e->lchild) En_LinkQueue(Tree, e->lchild);
		if (e->rchild) En_LinkQueue(Tree, e->rchild);
	};
};
int find(char Node, char* Tree)
{
	for (int i=0;i<strlen(Tree);i++)
		if (Node == Tree[i]) return i;
}
BiTree constructBitree(char* pPost, char* pMid, int iLen)
{
	if (iLen == 0) return NULL;
	BiTree pNode = new BiTNode;
	pNode->data = *(pPost + iLen - 1);
	int iPos=find(*(pPost + iLen - 1), pMid);
	pNode->lchild = constructBitree(pPost, pMid, iPos);
	pNode->rchild = constructBitree(pPost + iPos, pMid + iPos + 1, iLen - 1 - iPos);
	return pNode;
}
void CreatBiTree_ByInandPost()
{
	char pMid[MAXLEN], pPost[MAXLEN] = {'\0'};
	int iLen;
	BiTree T;
	cin >> pMid;
	cin >> pPost;
	iLen = strlen(pPost);
	T = constructBitree(pPost, pMid, iLen);
	PreOrderTraverse(T);
	cout << endl;
}
int main()
{
	BiTree Tree;
	CreatBiTree(Tree);
	PreOrderTraverse(Tree);
	cout << endl;
	InOrderTraverse(Tree);
	cout << endl;
	PostOrderTraverse(Tree);
	cout << endl;
	cout << NodeCount(Tree)<<endl;
	cout << NodeCount_Leaf(Tree)<<endl;
	LevelOrderTraverse(Tree);
	cout << endl;
	FreeBiTree(Tree);
	_CrtDumpMemoryLeaks();
	CreatBiTree_ByInandPost();
	return 0;
}
//ABD###CEG###FH##I##
//ABDCEGFHI
//DBAGECHFI
//DBGEHIFCA