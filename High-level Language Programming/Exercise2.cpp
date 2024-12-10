#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
int digui(int (*a)[5])
{
	static int k = 0;
	k++;
	printf("%d\n", k);
	return(0);
}
void main()
{
	int a[5][5];
	digui(a);
}
/*#include <stdio.h>
void main()
{
	struct text
	{
		int t;
		struct* text;
	};
	int a[2] = { 0,1 };
	int* p = a;
	printf("%d %d\n",p++,p);
}
#include <stdio.h>
struct T { int x; struct T* y; }data[2] = { 10, 0, 20, 0 };
void main()
{
	struct T* p = data;
	p->y = data;
	printf("%d\n",p++->x++);
	printf("%d\n", data[0].x);
}
// 插入排序 while循环实现
#include <stdio.h>
int main()
{
	int a[5] = { 4, 7, 2, 5, 1 };
	int i, j, m;
	for (i = 1; i < 5; i++)
	{
		m = a[i];
		j = i - 1;
		while (j >= 0 && m > a[j])
		{
			a[j + 1] = a[j];
			j--;
		}
		a[j + 1] = m;
	}
	for (i = 0; i < 5; i++)
		printf("%5d", a[i]);
	printf("\n");
	return 0;
}
// 测试程序9 关于不同类型的变量之间的运算
#include <stdio.h>
void main()
{
	printf("%f\n", 2 + 'a' - 1 + 1.0 / 2);
}
// 汉诺塔问题中关于第n部在第a个盘上 循环实现
#include <stdio.h>
void main()
{
	int a, b, t, ans;
	int wei[64] = { 0 };
	scanf_s("%d %d", &a, &b);
	while (a + b != 0)
	{
		if (a % 2 == 0)
		{
			for (int i = 1; i <= b; i++)
			{
				t = i;
				ans = 1;
				while (t % 2 == 0) {
					t /= 2;
					ans++;
				}
				if (i == b) printf("%d\n", ans);
				if (ans % 2 == 0) wei[ans] = (wei[ans] + 2) % 3; else wei[ans] = (wei[ans] + 1) % 3;
			}
		}
		else
		{
			for (int i = 1; i <= b; i++)
			{
				t = i;
				ans = 1;
				while (t % 2 == 0)
				{
					t /= 2; ans++;
				}
				if (i == b) printf("%d\n", ans);
				if (ans % 2 == 0) wei[ans] = (wei[ans] + 1) % 3; else wei[ans] = (wei[ans] + 2) % 3;
			}
		}
		scanf_s("%d %d", &a, &b);
		for (int i = 0; i < 64; i++)
			wei[i] = 0;
	}
}
// 汉诺塔问题中关于第n部在第a个盘上 递归实现
#include <stdio.h>
int num, ans;
void han(int n, char a, char b, char c)
{
	if (n == 1)
	{
		num++;
		if (num == ans)
		{
			printf("%c\n", a);
			return;
		}
	}
	else
	{
		han(n - 1, a, c, b);
		num++;
		if (num == ans)
		{
			printf("%c\n", a);
			return;
		}
		han(n - 1, b, a, c);
	}
}
void main()
{
	int a;
	scanf_s("%d %d", &a, &ans);
	while (a + ans != 0)
	{
		num = 0;
		han(a, '1', '2', '3');
		scanf_s("%d %d", &a, &ans);
	}
}
// 成绩处理 函数实现 输入、求平均、求最大、输出
#include <stdio.h>
int n = 10, m = 5;
int x;
int y;
void scan(float a[11][6])
{
	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= m; j++)
		{
			scanf_s("%f", &a[i][j]);
			a[0][j] += a[i][j];
			a[i][0] += a[i][j];
		}
}
void average(float a[11][6])
{
	for (int i = 1; i <= n; i++)
		a[i][0] /= m;
	for (int j = 1; j <= m; j++)
		a[0][j] /= n;
}
void print(float a[11][6])
{
	for (int i = 1; i <= n; i++)
		printf("%-7.2f", a[i][0]);
	printf("\n");
	for (int j = 1; j <= m; j++)
		printf("%-7.2f", a[0][j]);
	printf("\n");
}
void max(float a[11][6])
{
	for (int i = 1; i <= n; i++)
		for (int j = 1; j <= m; j++)
			int(a[x][y]) < int(a[i][j]) ? x = i, y = j : 0;
	printf("%5d%5d%5d\n", x, y, int(a[x][y]));
}
void main()
{
	float a[11][6] = { 0 };
	scan(a);
	average(a);
	print(a);
	x = 1;
	y = 1;
	max(a);
}
// 二分查找法
#include <stdio.h>
int a[15];
int aim;
int erfen(int L, int R)
{
	int M = (L + R) / 2;
	if (L > R) return -1;
	if (a[M] == aim) return M;
	if (a[M] < aim) return erfen(M + 1, R);
	return erfen(L, M - 1);
}
void main()
{
	for (int i = 0; i < 15; i++)
		scanf_s("%d", &a[i]);
	scanf_s("%d", &aim);
	if (erfen(0, 14) != -1) printf("%d\n", erfen(0, 14) + 1); else printf("Not find!\n");
}
// 寻找二维数组中的鞍点
#include <stdio.h>
void main()
{
	int a[4][5] = { 0 };
	int flag = 1;
	int max[4] = { 0 };
	int min[5] = { 0 };
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 5; j++)
		{
			scanf_s("%d", &a[i][j]);
			min[j] = a[min[j]][j] < a[i][j] ? min[j] : i;
			max[i] = a[i][max[i]] > a[i][j] ? max[i] : j;
		}
	for (int i = 0; i < 4; i++)
		if (min[max[i]] == i)
		{
			printf("a[%d][%d]=%d", i, max[i], a[i][max[i]]);
			flag = 0;
		}
	if (flag) printf("It's not exist!\n"); else printf("\n");
}
// 插入排序 for实现
#include <stdio.h>
void main()
{
	int a[11] = { 0 };
	int i, t, k = 0;
	for (i = 0; i < 10; i++)
		scanf_s("%d", &a[i]);
	scanf_s("%d", &a[10]);
	for (i = 0; i < 10; i++)
		if (a[i] >= a[10])
		{
			k = i;  break;
		}
	t = a[10];
	for (i = 10; i > k; i--)
		a[i] = a[i - 1];
	a[k] = t;
	for (i = 0; i <= 10; i++)
		printf("%5d", a[i]);
}
// 冒泡排序
#include<stdio.h>
void main()
{
	int i, j, k, a[10];
	for (i = 0; i < 10; i++)
		scanf_s("%d", &a[i]);
	printf("\n");
	for (i = 0; i < 9; i++)
		for (j = 0; j < 9 - i; j++)
			if (a[j] > a[j + 1])
			{
				k = a[j]; a[j] = a[j + 1]; a[j + 1] = k;
			}
	for (i = 0; i <= 9; i++)
		printf("%5d", a[i]);
	printf("\n");
}
// 汉诺塔问题 动态规划实现
#include <stdio.h>
#include <math.h>
void main()
{
	int n, t, k;
	int wei[64] = { 0 };
	scanf_s("%d", &n);
	for (int i = 1; i <= pow(2, n) - 1; i++)
	{
		k = i; t = 1;
		while (k % 2 == 0) { k /= 2; t++; }
		printf("%c-->", wei[t] + 65);
		wei[t] = (wei[t] + 2 - (n + t) % 2) % 3;
		printf("%c\n", wei[t] + 65);
	}
}
// 按要求求年龄 递归实现
#include <stdio.h>
int age(int n)
{
	if (n == 1) return 10;
	else return age(n - 1) + 2;
}
void main()
{
	printf("age: % d\n", age(5));
}
// 测试程序8 关于指针、sizeof和strlen区别
#include <stdio.h>
#include <string.h>
int main()
{
	char str[10] = { "I ha ppy" };
	char str1[] = "I am lei";
	const char* str2 = "I am lei";
	int a1 = sizeof(str);
	int a2 = strlen(str);

	int b1 = sizeof(str1);
	int b2 = strlen(str1);

	int c1 = sizeof(str2);
	int c2 = strlen(str2);

	printf("%d,%d,%d,%d,%d,%d\n", a1, a2, b1, b2, c1, c2);
	return 0;
}
// 爬楼梯 递归实现
#include <stdio.h>
int pa(int num)
{
	if (num < 0) return 0;
	if ((num == 1) || (num == 0))  return 1;
	if (num == 2) return 2;
	return(pa(num - 1) + pa(num - 2) + pa(num - 3));
}
void main()
{
	int n;
	scanf_s("%d", &n);
	printf("%d\n", pa(n));
}
// 完美数2
#include <stdio.h>
void main()
{
	int n, sum;
	for (n = 2; n < 1000; n++)
	{
		sum = 0;
		for (int i = 1; i < n; i++)
			if (n % i == 0) sum += i;
		if (sum == n)
		{
			printf("%d its factors are", n);
			for (int i = 1; i < n; i++)
				if (n % i == 0)
				{
					sum -= i;
					if (sum != 0) printf("%d,", i); else printf("%d\n", i);
				}
		}
	}
}
// 输出菱形
#include <stdio.h>
void main()
{
	int n = 1;
	for (int i = 1; i <= 7; i++)
	{
		for (int j = 1; j <= 7; j++)
			if ((j + (n - 1) / 2 <= 3) || (j - (n - 1) / 2 >= 5)) printf(" "); else printf("*");
		if (i < 4) n += 2; else n -= 2;
		printf("\n");
	}
}
// 找出最大字符串
#include <stdio.h>
#include <string.h>
void main()
{
	char str[3][20];
	char string[20];
	for (int i = 0; i < 3; i++)
		gets_s(str[i]);
	if (strcmp(str[0], str[1]) > 0) strcpy_s(string, str[0]); else strcpy_s(string, str[1]);
	if (strcmp(string, str[2]) < 0) strcpy_s(string, str[2]);
	printf("%s\n", string);
}
// 二维数组中寻找最大数
#include <stdio.h>
void main()
{
	int a[3][4] = { {1,2,3,4},{9,8,7,6},{-10,10,-5,2} };
	int max, row = 0, com = 0;
	max = a[0][0];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			if (a[i][j] > max)
			{
				max = a[i][j];
				row = i;
				com = j;
			}
	printf("%5d%5d%5d\n", max, row + 1, com + 1);
}
// 两个数组实现按列相加
#include <stdio.h>
void main()
{
	int a[3][4] = { {1,2,3,4},{5,6,7,8},{9,10,11,12} };
	int b[3][4] = { {13,14,15,16},{17,18,19,20},{21,22,23,24} };
	int ans[3][4] = { 0 };
	for (int j = 0; j < 4; j++)
		for (int i = 0; i < 3; i++)
			ans[i][j] = a[i][j] + b[i][j];
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 4; j++)
			printf("%5d", ans[i][j]);
		printf("\n");
	}
}
// 选择排序
#include <stdio.h>
void main()
{
	int a[11] = { 0 };
	int t, k;
	for (int i = 1; i <= 10; i++)
		scanf_s("%d", &a[i]);
	printf("\n");
	for (int i = 1; i <= 10; i++)
	{
		k = i;
		for (int j = i + 1; j <= 10; j++)
			if (a[k] > a[j]) k = j;
		t = a[i]; a[i] = a[k]; a[k] = t;
	}
	for (int i = 1; i <= 10; i++)
		printf("%d ", a[i]);
	printf("\n");
}
// 测试程序7 关于int和char之间的转换
#include <stdio.h>
void main()
{
	int a = 65;
	printf("%c\n", a);
}
// 测试程序6 关于运算顺序
#include <stdio.h>
void main()
{
	int a, b, m = 0, n = 0;
	a = 8;
	b = 10;
	m += a++;
	n -= --b;
	printf("a=%d,b=%d,m=%d,n=%d\n", a, b, m, n);
}
// 测试程序5 关于自增符
#include <stdio.h>
void main()
{
	int a = 10, b = 10;
	//printf("%d %d", a++, b++);
	printf("%d %d", ++a, ++b);
}
// 分数求和3
#include <stdio.h>
void main()
{
	double sum = 0, n, m, t;
	m = 2;
	n = 1;
	for (int i = 1; i <= 20; i++)
	{
		sum += (m / n);
		t = m;
		m += n;
		n = t;
	}
	printf("%lf\n", sum);
}
// 统计各类符号数量
#include <stdio.h>
void main()
{
	char c;
	int zimu = 0, shuzi = 0, kong = 0, other = 0;
	while ((c = getchar()) != '\n')
	{
		if (((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'Z'))) zimu++;
		else if ((c >= '0') && (c <= '9')) shuzi++;
		else if (c == ' ') kong++;
		else other++;
	}
	printf("%d %d %d %d\n", zimu, shuzi, kong, other);
}
// 程序实现分段函数计算
#include <stdio.h>
void main()
{
	double x;
	scanf_s("%lf", &x);
	if (x < 1.0) printf("y=%lf\n", x);
	else if (x < 10.0) printf("y=%lf\n", 2.0 * x - 1.0);
	else printf("y=%lf\n", 3.0 * x - 11.0);
}
// 三个数按从大到小输出
#include <stdio.h>
void main()
{
	int a, b, c, max;
	scanf_s("%d%d%d", &a, &b, &c);
	max = (a > b) ? a : b;
	max = (max > c) ? max : c;
	if (max == a) printf("%d %d %d\n", max, (b > c) ? b : c, (b < c) ? b : c);
	else if (max == b)  printf("%d %d %d\n", max, (a > c) ? a : c, (a < c) ? a : c);
	else printf("%d %d %d\n", max, (b > a) ? b : a, (b < a) ? b : a);
}
// 测试程序4 关于struct
#include <stdio.h>			
#include <windows.h>
struct player
{
	int x;
	int y;
};
char a;
short b;
int c;
void main()
{
	a = 1;
	b = 2;
	c = 3;
	player p = { 10,20 };
	printf("%x %x %x %x \n", &a, &b, &c, &p);
	return;
}
// 测试程序3 关于*
#include <stdio.h>
	// %p　八位十六进制大写　地址输出
	//　函数内部交换　不污染数据
void Exchg1(int x, int y)
{
	int tmp;
	tmp = x;
	x = y;
	y = tmp;
	printf("Exchg1:x=%d,y=%d\n", x, y);
}
	// 通过取地址运算实现数据交换
void Exchg2(int& x, int& y)
{
	int tmp;
	tmp = x;
	x = y;
	y = tmp;
	printf("Exchg2:x=%d,y=%d\n", x, y);
}
	// 通过指针运算实现数据交换
void Exchg3(int* x, int* y)
{
	int tmp;
	tmp = *x;
	*x = *y;
	*y = tmp;
	printf("Exchg3:x=%d,y=%d\n", *x, *y);
}
void main()
{
	int a = 4, b = 6;
	Exchg1(a, b);
	printf("a=%d,b=%d\n", a, b);
	a = 4; b = 6;
	Exchg2(a, b);
	printf("a=%d,b=%d\n", a, b);
	a = 4; b = 6;
	Exchg3(&a, &b);
	printf("a=%d,b=%d\n", a, b);
}
// 测试程序2 关于&
#include <stdio.h>
void main()
{
	int a = 10;
	printf("%d\n", a);
	printf("%p\n", a);
	printf("%d\n", &a);
	printf("%p\n", &a);
}
// 测试程序1 关于break
#include <stdio.h>
#include <math.h>
void main()
{
	double x, y, z;
	scanf_s("%LF%LF", &x, &y);
	z = x / y;
	while (1)
		if (fabs(z) > 1.0) { x = y; y = x; z = x / y; }
		else break;
	printf("y=%f\n", y);
}
// 数组实现斐波那契数列
#include <stdio.h>
int main()
{
	int a[21] = { 0,1 };
	int i;
	for (i = 2; i <= 20; i++)
		a[i] = a[i - 1] + a[i - 2];
	for (i = 1; i <= 20; i++)
		printf("%6d", a[i]);
	printf("\n");
	return 0;
}
// 通过数列通项的方法求斐波那契数列
#include <stdio.h>
#include <math.h>
void main()
{
	unsigned long long int a[81] = { 0 };
	for (int i = 1; i <= 80; i++)
		printf("%20llu", a[i] = unsigned long long int((pow((1.0 + sqrt(5.0)) / 2.0, i) - pow((1.0 - sqrt(5.0)) / 2.0, i)) / sqrt(5.0)));
}
// 完美数1
# include <stdio.h>
void main()
{
	int m, s, i;
	for (m = 2; m < 500000; m++)
	{
		s = 0;
		for (i = 1; i < m; i++)
		{
			if ((m % i) == 0)  s += i;
			if (s > m) break;
		}
		if (s == m) printf("%d\n", m);
	}
}
// 100-200间的素数
#include <stdio.h>
#include <math.h>
void main()
{
	int n, i, m = 0;
	for (n = 101; n <= 200; n += 2)
	{
		for (i = 3; i <= int(sqrt(n)); i++)
			if (n % i == 0) break;
		if (i >= int(sqrt(n)) + 1)
		{
			printf("%d ", n);
			m++;
		}
		if (m % 10 == 0) printf("\n");
	}
	printf("\n");
}
// double类型的测试
#include <stdio.h>
void main()
{
	int s1 = 0, s2 = 0;
	float s3 = 0.0;
	for (int k = 1; k <= 100; k++)
		s1 += k;
	for (int k = 1; k <= 50; k++)
		s2 += k * k;
	for (double k = 1; k <= 10; k++)
		s3 += 1 / k;
	s3 += s1 + s2;
	s3 = 1.0 / 3.0;
	printf("%.16f\n", s3);
}
// 水仙花数
#include <math.h>
#include <stdio.h>
void main()
{
	int i, j, k, n;
	for (n = 100; n < 1000; n++)
	{
		i = n / 100;
		j = (n / 10) % 10;
		k = n % 10;
		if (n == pow(i, 3) + pow(j, 3) + pow(k, 3)) printf("%d\t", n);
	}
	printf("\n");
}
// 程序实现 n+nn+nnn的计算
#include <stdio.h>
void main()
{
	int a, n, sum = 0;
	scanf_s("%d%d", &a, &n);
	for (int i = 1; i <= n; i++)
	{
		sum += a;
		a = a % 10 + a * 10;
	}
	printf("%d,%d\n", sum, a);
}
// 二分法求方程解
#include <stdio.h>
#include <math.h>
void main()
{
	float x0, x1, x2, fx0, fx1, fx2;
	x1 = -10; fx1 = 2 * pow(x1, 3) - 4 * pow(x1, 2) + 3 * x1 - 6;
	x2 = 10; fx2 = 2 * pow(x2, 3) - 4 * pow(x2, 2) + 3 * x2 - 6;
	do
	{
		x0 = (x1 + x2) / 2;
		fx0 = 2 * pow(x0, 3) - 4 * pow(x0, 2) + 3 * x0 - 6;
		if (fx0 * fx1 < 0)
		{
			x2 = x0;
			fx2 = fx0;
		}
		else
		{
			x1 = x0;
			fx1 = fx0;
		}
	} while (fabs(fx0) >= 1e-5);
	printf("x=%6.2f\n", x0);
}
// 分数求和2
#include <stdio.h>
void main()
{
	int i;
	double a = 2, b = 1, s = 0, t;
	for (i = 1; i <= 20; i++)
	{
		s += a / b;
		t = a;
		a += b;
		b = t;
	}
	printf("sum=%16.10f\n", s);
}
// 求π
#include <stdio.h>
#include <math.h>
void main()
{
	int sign = 1;
	double pi = 0.0, n = 1.0, term = 1.0;
	while (fabs(term) >= 1e-6)
	{
		pi += term;
		n += 2;
		sign = -sign;
		term = sign / n;
	}
	pi *= 4;
	printf("pi=%10.8f\n", pi);
	printf("%d\n", (int(n) - 1) / 2);
}
// 矩阵输出
#include <stdio.h>
void main()
{
	int n = 0;
	for (int i = 1; i <= 4; i++)
		for (int j = 1; j <= 5; j++, n++)
		{
			if (n % 5 == 0) printf("\n");
			printf("%d\t", i * j);
		}
	printf("\n");
}
// 在数组中寻找一个数
#include <stdio.h>
int main()
{
	int n, m, i, a[100000] = { 0 };
	scanf_s("%d", &n);
	for (i = 1; i <= n; i++)
		scanf_s("%d", &a[i]);
	scanf_s("%d", &m);
	for (i = 1; i <= n; i++)
	{
		if (a[i] == m)
		{
			printf("%d", i);
			break;
		}
	}
	if (i == n + 1) printf("Not Exist!");
	return 0;
}
//循环求和1
#include <stdio.h>
void main()
{
	int i = 1, sum = 0;
loop: if (i <= 200)
{
	sum += i;
	i++;
	goto loop;
}
printf("%d", sum);
}
//循环求和2
#include <stdio.h>
void main()
{
	int sum = 0;
	for (int i = 1; i <= 100; i++)
		sum += i;
	printf("%d", sum);
}
// 机器猫题目
#include<stdio.h>
void main()
{
	int target, step;
	scanf_s("%d", &target);
	step = 0;
	while (target != 1)
	{
		if (target % 2 == 0) { target = target / 2; step++; }
		else if (((step != 0) && ((target + 1) % 4 == 0)) && (target != 3)) { target++; step++; }
		else { target--; step++; }
	}
	printf("%d", step);
}
// 分数求和1
#include <stdio.h>
void main()
{
	float sum, i, k;
	k = 1;
	sum = 0;
	for (i = 1; i <= 100; i++)
	{
		sum = sum + k * 1 / i;
		k = -k;
	}
	printf("%f\n", sum);
}
// 判断闰年
#include <stdio.h>
void main()
{
    int a;
    scanf_s("%d", &a);
    if (a % 4 != 0) printf("NO\n");
    else if (a % 100 != 0) printf("Yes\n");
    else if (a % 400 == 0) printf("Yes\n");
    else printf("No\n");
}
// 输出hello,world
#include <stdio.h>
int main()
{
	printf("hello,world!\n");
	return 0;
}
*/