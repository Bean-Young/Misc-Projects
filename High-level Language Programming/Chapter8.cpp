//15.
#include <stdio.h>
#define M 4
#define N 5
float aver[M];
int main()
{
	int i, j;
	int id[M];
	float score[M][N], (*p)[N];        //（*p）要按列取
	void aver1(float(*p)[N]);
	void less60(float(*p)[N], int id[]);
	void best(float(*p)[N], int id[]);

	for (i = 0; i < M; i++)
	{
		printf("\n请第%d个学生的学号：", i + 1);
		scanf("%d", &id[i]);
		printf("\n请第%d个学生的成绩：", i + 1);
		for (j = 0; j < N; j++)
			scanf("%f", &score[i][j]);
	}


	p = score;

	aver1(p);
	less60(p, id);
	best(p, id);

	return 0;
}
void aver1(float(*p)[N])
{
	int i;
	float aver, sum = 0;
	for (i = 0; i < M; i++)
		sum += *(*(p + i));
	aver = sum / 4.0;
	printf("第一门课的平均分是：%.2f\n", aver);
}
void less60(float(*p)[N], int id[M])
{
	int i, j, n;
	float sum;
	for (i = 0; i < M; i++)
	{
		sum = 0;
		for (j = 0; j < N; j++)
			sum += *(*(p + i) + j);
		aver[i] = sum / 5.0;
	}
	for (i = 0; i < M; i++)
	{
		int flag = 0;
		for (j = 0; j < N; j++)
		{
			if (*(*(p + i) + j) < 60.0)
				flag++;
		}
		if (flag > 2)
		{
			printf("学号为%d的同学有两门以上成绩不及格！其全部成绩及平均成绩如下：\n", id[i]);
			for (n = 0; n < N; n++)
				printf("%.2f ", *(*(p + i) + n));
			printf("\n");
			printf("平均成绩：%.2f\n", aver[i]);
		}
	}
}
void best(float(*p)[N], int id[M])
{
	int i, j, flag;
	for (i = 0; i < M; i++)
	{
		flag = 1;
		if (aver[i] <= 90.0)
			flag = 0;
		for (j = 0; j < N; j++)
			if (*(*(p + i) + j) <= 85.0)
				flag = 0;
		if (flag == 1)
			printf("学号为%d的同学平均成绩在90分以上或全部课程成绩在85分以上\n", id[i]);
	}
}
//14.
#include <stdio.h>
void exchange(int* p, int len)
{
	for (int i = 0; i < len / 2; i++)
	{
		int t = *(p + i);
		*(p + i) = *(p + len - 1 - i);
		*(p + len - 1 - i) = t;
	}
}
void main()
{
	int n;
	int a[20];
	scanf_s("%d", &n);
	for (int i = 0; i < n; i++)
		scanf_s("%d", &a[i]);
	exchange(a, n);
	for (int i = 0; i < n; i++)
		printf("%5d", a[i]);
	printf("\n");
}
//16.
#include <stdio.h>
#include<string.h>
int main()
{
	int i, j = 0, n = 0;
	int len, cnt = 0, a[1000] = { 0 }, tmp = 0;
	char str[1000] = { 0 }, * p, temp[1000] = { 0 };

	printf("请输入字符串：");
	gets_s(str);
	len = strlen(str);
	p = str;

	for (i = 0; i <= len; i++)
	{

		if (*(p + i) >= '0' && *(p + i) <= '9')
		{
			cnt++;
			temp[j++] = *(p + i);
		}
		else
		{
			int x = 1;
			if (cnt != 0)
			{
				j = cnt;
				for (; cnt > 0; cnt--)
				{
					tmp += (temp[j - 1] - '0') * x;
					x *= 10;
					j--;
				}
				a[n++] = tmp;
			}
			j = 0;
			tmp = 0;
			cnt = 0;
		}
	}

	printf("整数个数是%d\n", n);
	for (i = 0; i < n; i++)
		printf("%d ", a[i]);
	printf("\n");
	return 0;
}
//21.
#include <stdio.h>
void sort(int** p, int n)
{
	int* t;
	for (int i = 0; i < n - 1; i++)
		for (int j = i + 1; j < n; j++)
			if (**(p + i) > **(p + j))
			{
				t = *(p + i); *(p + i) = *(p + j); *(p + j) = t;
			}
}
void main()
{
	int n, data[20], ** p, * ps[20];
	scanf_s("%d", &n);
	for (int i = 0; i < n; i++)
	{
		ps[i] = &data[i];
		scanf_s("%d", ps[i]);
	}
	p = ps;
	sort(p, n);
	for (int i = 0; i < n; i++)
		printf("%5d", *ps[i]);
	printf("\n");
}
//20.
#include <stdio.h>
#include <string.h>
void sort(char** p)
{
	char* t;
	for (int i = 0; i < 5; i++)
		for (int j = i + 1; j < 5; j++)
		{
			if (strcmp(*(p + i), *(p + j)) > 0)
			{
				t = *(p + i); *(p + i) = *(p + j); *(p + j) = t;
			}
		}
}
void main()
{
	char** p, * ps[5], str[5][20];
	for (int i = 0; i < 5; i++)
		ps[i] = str[i];
	for (int i = 0; i < 5; i++)
		gets_s(str[i]);
	p = ps;
	sort(p);
	for (int i = 0; i < 5; i++)
		printf("%s\n", ps[i]);
}
/*
//14.
#include <stdio.h>
void exchange(int* p, int len)
{
	for (int i = 0; i < len / 2; i++)
	{
		int t = *(p + i);
		*(p + i) = *(p + len - 1 - i);
		*(p + len - 1 - i) = t;
	}
}
void main()
{
	int n;
	int a[20];
	scanf_s("%d", &n);
	for (int i = 0; i < n; i++)
		scanf_s("%d", &a[i]);
	exchange(a, n);
	for (int i = 0; i < n; i++)
		printf("%5d", a[i]);
	printf("\n");
}
//10.
#include <stdio.h>
void exchange(int* p)
{
	int t;
	int* pmax, * pmin;
	pmax = pmin = p;
	for (int i = 0; i < 5; i++)
		for (int j = i; j < 5; j++)
		{
			pmax = *pmax > *(p + 5 * i + j) ? pmax : (p + 5 * i + j);
			pmin = *pmin < *(p + i * 5 + j) ? pmin : (p + 5 * i + j);
		}
	t = *(p + 12); *(p + 12) = *pmax; *pmax = t;
	t = *p; *p = *pmin; *pmin = t;
	pmin = p + 1;
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 5; j++)
			if ((p + 5 * i + j) != p) pmin = *pmin < *(p + 5 * i + j) ? pmin : (p + 5 * i + j);
	t = *pmin; *pmin = *(p + 4); *(p + 4) = t;
	pmin = p + 1;
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 5; j++)
			if (((p + 5 * i + j) != (p + 4)) && ((p + 5 * i + j) != p)) pmin = *pmin < *(p + 5 * i + j) ? pmin : (p + 5 * i + j);
	t = *pmin; *pmin = *(p + 20); *(p + 20) = t;
	pmin = p + 1;
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 5; j++)
			if (((p + 5 * i + j) != (p + 4)) && ((p + 5 * i + j) != p) && ((p + 5 * i + j) != (p + 20))) pmin = *pmin < *(p + 5 * i + j) ? pmin : (p + 5 * i + j);
	t = *pmin; *pmin = *(p + 24); *(p + 24) = t;
}
void main()
{
	int a[5][5], * p;
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 5; j++)
			scanf_s("%d", &a[i][j]);
	exchange(a[0]);
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
			printf("%4d", a[i][j]);
		printf("\n");
	}
}
//9.
#include <stdio.h>
void exchange(int* p)
{
	for (int i = 0; i < 3; i++)
		//j从i开始
		for (int j = i; j < 3; j++)
		{
			int t = *(p + i * 3 + j);
			*(p + i * 3 + j) = *(p + j * 3 + i);
			*(p + j * 3 + i) = t;
		}
}
void main()
{
	int a[3][3];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			scanf_s("%d", &a[i][j]);
	exchange(a[0]);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			printf("%3d", a[i][j]);
		printf("\n");
	}
}
//7.
#include <stdio.h>
#include <string.h>
void copy(char* p1, char* p2, int m)
{
	int n = 0;
	while (n < m - 1)
	{
		n++;
		p1++;
	}
	while (*p1 != '\0')
	{
		*p2 = *p1;
		p1++;
		p2++;
	}
	*p2 = '\0';
}
void main()
{
	int m;
	char str1[20], str2[20];
	gets_s(str1);
	scanf_s("%d", &m);
	if (strlen(str1) < m)
		printf("input error!");
	else
	{
		copy(str1, str2, m);
		printf("%s\n", str2);
	}
}
//6.
#include <stdio.h>
int strlen_1(char* s)
{
	int n = 0;
	while (*s != '\0')
	{
		s++;
		n++;
	}
	return(n);
}
void main()
{
	char str[20];
	gets_s(str);
	printf("%d\n", strlen_1(str));
}
//5.
#include <stdio.h>
void main()
{
	int n, num[50], * p, flag, exit, now;
	now = exit = flag = 0;
	p = num;
	scanf_s("%d", &n);
	for (int i = 0; i < n; i++)
		*(p + i) = i + 1;
	while (exit < n - 1)
	{
		if (*(p + now) != 0) flag++;
		if (flag == 3)
		{
			*(p + now) = 0;
			flag = 0;
			exit++;
		}
		now++;
		if (now == n) now = 0;
		//num数组存放编号 num[now]=0即为退出 num数组下标为0~n-1
	}
	while (*(p++) == 0);
	printf("%d\n", *(--p));
}
//3.
#include <stdio.h>
void scan(int* a)
{
	for (int i = 0; i < 10; i++)
		scanf_s("%d", (a + i));
}
void print(int* a)
{
	for (int i = 0; i < 10; i++)
		printf("%5d", *(a + i));
	printf("\n");
}
void exchange(int* a)
{
	int* max, * min, t;
	max = min = a;
	for (int i = 0; i < 10; i++)
	{
		max = *max > *(a + i) ? max : a + i;
		min = *min < *(a + i) ? min : a + i;
	}
	t = *a; *a = *min; *min = t;
	if (max == a) max = min;
	t = *(a + 9); *(a + 9) = *max; *max = t;
}
void main()
{
	int a[10];
	scan(a);
	exchange(a);
	print(a);
}
//2.
//strcpy_s(char*s1,int len,char*s2); s2前len个放入s1
#include <stdio.h>
#include <string.h>
void exchange(char* p, char* q)
{
	char t[20];
	if (strcmp(p, q) > 0) { strcpy_s(t, strlen(p) + 1, p); strcpy_s(p, strlen(q) + 1, q); strcpy_s(q, strlen(t) + 1, t); }
}
void main()
{
	char a[20], b[20], c[20];
	gets_s(a);
	gets_s(b);
	gets_s(c);
	exchange(a, b);
	exchange(a, c);
	exchange(b, c);
	printf("%s\n%s\n%s\n", a, b, c);
}
//1.
#include <stdio.h>
void exchange(int* p, int* q)
{
	if (*p > *q) { int t = *p; *p = *q; *q = t; }
}
void main()
{
	int a, b, c;
	scanf_s("%d%d%d", &a, &b, &c);
	exchange(&a, &b);
	exchange(&a, &c);
	exchange(&b, &c);
	printf("%5d%5d%5d\n", a, b, c);
}
*/