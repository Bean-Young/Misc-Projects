#include "stdafx.h"
using namespace std;
#define OK 1
#define ERROR 0
#define OVERFLOW -2
#define MVNum 100
#define MAXINT 32767
typedef int Status;
typedef int VerTexType;
typedef int OtherInfo;
typedef struct ArcNode
{
	int adjvex;
	struct ArcNode* nextarc;
	OtherInfo info;
};
typedef struct VNode
{
	VerTexType data;
	ArcNode* firstarc;
}VNode, AdjList[MVNum];
typedef struct
{
	AdjList vertices;
	int vexnum, arcnum;
}ALGraph;
int LocateVex_AL(ALGraph G, int v)
{
	for (int i = 0; i < G.vexnum; i++)
		if (G.vertices[i].data == v) return i;
	return MAXINT;
}
Status CreatUDG(ALGraph& G)
{
	cin >> G.vexnum >> G.arcnum;
	for (int i = 0; i < G.vexnum; i++)
	{
		cin >> G.vertices[i].data;
		G.vertices[i].firstarc = NULL;
	}
	for (int k = 0; k < G.arcnum; k++)
	{
		int v1, v2, w, i, j;
		cin >> v1 >> v2>>w;
		i = LocateVex_AL(G, v1);
		j = LocateVex_AL(G, v2);
		ArcNode* p1 = new ArcNode;
		p1->adjvex = j;
		p1->info = w;
		p1->nextarc = G.vertices[i].firstarc;
		G.vertices[i].firstarc = p1;
		ArcNode * p2= new ArcNode;
		p2->adjvex=i;
		p2->info=w;
		p2->nextarc=G.vertices[j].firstarc;
		G.vertices[j].firstarc=p2;
	}
	return OK;
}
void Print_ALGraph(ALGraph G)
{
	for (int i = 0; i < G.vexnum; i++)
	{
		cout << G.vertices[i].data << endl;
		ArcNode* p;
		p = G.vertices[i].firstarc;
		while (p)
		{
			cout << G.vertices[i].data << "->" << G.vertices[p->adjvex].data << ' ' << p->info << endl;
			p = p->nextarc;
		}
	}
}
int main()
{
	ALGraph ALG;
	CreatUDG(ALG);
	Print_ALGraph(ALG);
	return 0;
}