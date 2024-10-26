#include <iostream>
#include <string>
using namespace std;
#define OK 1
#define ERROR 0
#define OVERFLOW -2
#define MAXINT 32767
#pragma warning (disable:4996)
typedef struct {
	int weight;
	int parent, lchild, rchild;
}HTNode,*HuffmanTree;
typedef char** HuffmanCode;
void Select(HuffmanTree HT, int n, int& i, int& j)
{
	i = j = 0;
	for (int k = 1; k <= n; k++)
		if ((HT[k].parent==0)&&(HT[k].weight < HT[i].weight))
			i = k;
	for (int k = 1; k <= n; k++)
		if ((HT[k].parent == 0) && (HT[k].weight < HT[j].weight)&&(k!=i))
			j = k;
}
void InitHuffmanTree(HuffmanTree& HT, int n)
{
	if (n <= 1) return;
	int m = 2 * n - 1;
	HT = new HTNode[m + 1];
	HT[0].weight = MAXINT;
	for (int i = 1; i <= m; i++)
	{
		HT[i].parent = 0;
		HT[i].lchild = 0;
		HT[i].rchild = 0;
	}
}
void CreatHuffmanTree(HuffmanTree& HT, int n)
{
	int m = 2 * n - 1;
	int s1, s2;
	for (int i = n + 1; i <= m; i++)
	{
		Select(HT, i - 1, s1, s2);
		HT[s1].parent = i;
		HT[s2].parent = i;
		HT[i].lchild = s1;
		HT[i].rchild = s2;
		HT[i].weight = HT[s1].weight + HT[s2].weight;
	}
}
void CreatHuffmanCode(HuffmanTree HT, HuffmanCode& HC, int n)
{
	HC = new char* [n + 1];
	char * cd = new char[n];
	cd[n - 1] = '\0';
	for (int i = 1; i <= n; i++)
	{
		int start = n - 1; //层数 n-start
		int c = i;
		int f = HT[i].parent;
		while (f)
		{
			start--;
			if (HT[f].lchild == c) cd[start] = '0';
			else cd[start] = '1';
			c = f;
			f = HT[f].parent;
		}
		HC[i] = new char[n - start];
		strcpy(HC[i], &cd[start]);
	}
	delete cd;
}
void PrintCode(HuffmanCode HC, int n)
{
	for (int i = 1; i <= n; i++)
		cout << HC[i] << endl;
}
void PrintCodeString(HuffmanCode HC, int n, int* tong)
{
	int i = 1;
	char c;
	for (int j = 0; j < 27; j++)
	{
		if (tong[j])
		{
			if (j != 26)
			{
				c = 'a' + j;
				cout << c << '\t' << HC[i] << endl;
			}
			else
				cout << ' ' << '\t' << HC[i] << endl;
			i++;
		}
	}
}
void Code()
{
	char c='\0';
	int tong[27] = { 0 };
	getchar();
	while ((c=getchar())!='\n')
		if (c != ' ') tong[c - 'a']++; else tong[26]++;
	int n=0;	
	HuffmanTree HT;
	for (int i = 0; i < 27; i++)
		if (tong[i]) n++;
	InitHuffmanTree(HT, n);
	int k = 1;
	for (int i = 0; i < 27; i++)
		if (tong[i])
		{
			HT[k].weight = tong[i];
			k++;
		}
	CreatHuffmanTree(HT, n);
	HuffmanCode HC;
	CreatHuffmanCode(HT, HC, n);
	PrintCodeString(HC, n,tong);
}
void PrintString(HuffmanCode HC, int n, FILE* p)
{
	char c = '\0';
	while ((c = fgetc(p)) != EOF)
		cout << HC[c];
}
void File()
{
	FILE* p = fopen("a.text", "r");
	char c = '\0';
	int tong[256];
	HuffmanTree HT;	
	while ((c = fgetc(p)) != EOF)
		tong[c]++;
	int n = 0;	
	for (int i = 0; i < 256; i++)
		if (tong[i]) n++;
	InitHuffmanTree(HT, n);
	int k = 1;
	for (int i = 0; i < 27; i++)
		if (tong[i])
		{
			HT[k].weight = tong[i];
			k++;
		}
	CreatHuffmanTree(HT, n);
	HuffmanCode HC;
	CreatHuffmanCode(HT, HC, n);
	//PrintString(HC, n,p);
	fclose(p);
}
int main()
{
	HuffmanTree HT;
	HuffmanCode HC;
	int n;
	cin >> n;
	InitHuffmanTree(HT, n);
	for (int i = 1; i <= n; i++)
		cin >> HT[i].weight;
	CreatHuffmanTree(HT, n);
	CreatHuffmanCode(HT, HC, n);
	PrintCode(HC, n);
	Code();
	File();
	return 0;
}