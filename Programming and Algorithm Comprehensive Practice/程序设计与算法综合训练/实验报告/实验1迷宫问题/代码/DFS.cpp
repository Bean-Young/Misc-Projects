#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#define maxn 1000
using namespace std;
int n = 4, m = 4;//n*m迷宫 
int flag = 0; //若有能到终点的路，则flag变为1
int dir[4][2] = { {1,0},{-1,0},{0,1},{0,-1} }; //方向数组 ,分别为下，上，右，左 ，输出分别设为0,1,2,3
int vis[maxn][maxn];//DFS标记数组 

int mp[maxn][maxn] = {
	{1, 1, 1, 1, 1, 1},
	{1, 0, 1, 1, 1, 1},
	{1, 0, 0, 0, 0, 1},
	{1, 0, 1, 0, 1, 1},
	{1, 0, 1, 0, 0, 1},
	{1, 1, 1, 1, 1, 1}
};  //输入的迷宫 
typedef struct Point
{
	int x, y;
}SElemType;
SElemType solution[maxn * maxn];//方案记录数组

int Check(SElemType u)  //检查点是否有障碍和是否已经遍历过 
{
	if (mp[u.x][u.y] == 0 && vis[u.x][u.y] == 0)
		return 1;

	return 0;
}

void Output() //输出迷宫 
{
	printf("迷宫为：\n\n");
	for (int i = 1; i <= n; ++i)
	{
		for (int j = 1; j <= m; ++j)
			cout << mp[i][j];
		cout << endl;
	}
	cout << endl;
}

int Direction(SElemType a, SElemType b)  //节点a->b的方向（方向数组做了解释） int dir[4][2]={{1,0},{-1,0},{0,1},{0,-1}}; //方向数组 ,分别为下，上，右，左 ，输出分别设为0,1,2,3
{
	for (int i = 0; i < 4; ++i)
		if (b.x == a.x + dir[i][0] && b.y == a.y + dir[i][1])
			return i;
}


void Output2(int k)
{
	for (int i = 1; i < k; i++)
	{
		printf("(%d,%d,%d)\n", solution[i].x, solution[i].y, Direction(solution[i], solution[i + 1]));
	}
	printf("(%d,%d,OK)\n\n", n, m);//到达终点
}

void DFS(int k, int inx, int iny, int outx, int outy) {
	/*k表示当前走到第几步，x，y表示当前的位置*/
	solution[k].x = inx;
	solution[k].y = iny;
	vis[inx][iny] = 1;
	if ((inx == outx) && (iny==outy))
	{
		Output2(k);//如果到了终点就输出此方案
		flag = 1;
	}
	else
		for (int i = 0; i < 4; ++i)//四个方向遍历(下上右左)
		{
			int u = inx + dir[i][0], v = iny + dir[i][1];
			SElemType temp = { u, v };
			if (!Check(temp)) continue;//如果不是障碍1 就继续循环 是障碍1就 忽略 
			DFS(k + 1, u, v, outx, outy);
		}
}
void SolveDFS(int a, int b, int c, int d)
{
	cout << "项目2：递归输出：" << endl;
	flag = 0;
	//数组清空函数 memset：对于vis数组，sizeof(vis)个元素全部替换为第二个参数0
	memset(vis, 0, sizeof(vis));
	DFS(1, a, b, c, d);
	if (!flag) cout << "无通路,请重新输入" << endl;
}

int main() {
	int a = 1, b = 1, c = 4, d = 4;
	// 由起点(a,b)到终点(c,d)
	//Output 函数的作用是输出迷宫的初始化
	Output();
	
	SolveDFS(a, b, c, d);
	cout << "xxxxxxxxxxxxxxxxxxxxxx" << endl;

	getchar();
	return 0;
}
