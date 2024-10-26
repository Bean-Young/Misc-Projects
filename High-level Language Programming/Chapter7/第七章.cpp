//17.
#include <stdio.h>
void next(int n)
{
	int i;
	i = n / 10;
	if (i != 0) next(i);
	putchar(n % 10 + '0');
	putchar(32);
}
void main()
{
	int num;
	scanf_s("%d",&num);
	if (num < 0)
	{
		putchar('-');
		putchar(' ');
		num = -num;
	}
	next(num);
	printf("\n");
}
//16.
#include <stdio.h>
#include <math.h>
#include <string.h>
char s[100];
int change(char c)
{
	if ((c >='0') && (c <= '9')) return(c - 48); else return(c - 55);
}
int shi(int flag)
{
	if (flag == 0) return(change(s[flag]) * pow(16, (strlen(s) - 1)));
	return(change(s[flag]) * pow(16, (strlen(s) - 1 - flag)) + shi(flag - 1));
}
void main()
{
	gets_s(s);
	printf("%d\n", shi(strlen(s)-1));
}
//14.
#include <stdio.h>
#include <math.h>
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
void fangcha(float a[11][6])
{
	float sum = 0;
	float fc = 0;
	for (int i = 1; i <= n; i++)
		sum += a[i][0];
	for (int i = 1; i <= n; i++)
		fc += (pow(a[i][0], 2) - pow(sum / n, 2));
	printf("%6.2f\n", fc / n);
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
	fangcha(a);
}
//13.
#include <stdio.h>
float pn(int n,int x)
{
	if (n == 0) return 1;
	if (n == 1) return x;
	return((2 * n - 1) * x - pn((n - 1), x) - (n - 1) * pn((n - 2), x)) / n;
}
void main()
{
	int n,x;
	scanf_s("%d%d", &n,&x);
	printf("%6.2f\n", pn(n, x));
}


//10.
	//利用指针使函数返回多个值
#include <stdio.h>
void find_longest(char str[],int *b,int *e)
{
	int max = 0, begin = 0;
	for (int i = 0; str[i] != '\0'; i++)
	if (!(((str[i]>='a')&&(str[i]<='z')) || ((str[i]>='A')&&(str[i]<='Z'))))
	{
		if (max < (i - begin))
		{
			max = i - begin;
			*b = begin;
			*e = i - 1;
		}
		begin = i + 1;
	}
}
void main()
{
	char s[100];
	int begin, end;
	gets_s(s);
	find_longest(s, &begin, &end);
	for (int i = begin; i <= end; i++)
		printf("%c", s[i]);
	printf("\n");
}
//8.
#include <stdio.h>
void print(char s[])
{
	for (int i = 0; s[i] != '\0'; i++)
	{
		printf("%c", s[i]);
		if (s[i + 1] == '\0') printf("\n"); else printf(" ");
	}
}
void main()
{
	char str[100];
	gets_s(str);
	print(str);
}
//7.
#include <stdio.h>
void vowel(char str[], char str1[])
{
	int j = 0;
	for (int i = 0; str[i] != '\0'; i++)
		if ((str[i] == 'a') || (str[i] == 'e') || (str[i] == 'i') || (str[i] == 'o') || (str[i] == 'u') ||
			(str[i] == 'A') || (str[i] == 'E') || (str[i] == 'I') || (str[i] == 'O') || (str[i] == 'U')) str1[j++] = str[i];
}
void main()
{
	char s[100], s1[100] = { '\0' };
	gets_s(s);
	vowel(s, s1);
	printf("%s\n", s1);
}
//6.
#include <stdio.h>
void connect(char str1[], char str2[], char str[])
{
	int i, j;
	for (i = 0; str1[i] != '\0'; i++)
		str[i] = str1[i];
	for (j = 0; str2[j] != '\0'; j++)
		str[j + i] = str2[j];
}
void main()
{
	char s1[100], s2[100], s[100] = { '\0' };
	gets_s(s1);
	gets_s(s2);
	connect(s1, s2, s);
	printf("%s\n", s);
}
//5.
	//extern 用于全局变量
#include <string.h>
#include <stdio.h>
char str[100] = { '\0' };
void jh(char* p, char* q)
{
	char t;
	t = *p;
	*p = *q;
	*q = t;
}
void exchang_for_string()
{
	for ( int i = 0 ,j = strlen(str); i < j; i++, j--)
		jh(&str[i], &str[j - 1]);
}
void main()
{
	gets_s(str);
	exchang_for_string();
	printf("%s\n", str);
}
//4.
#include <stdio.h>
void jh(int* p,int *q)
{
	int t;
	t = *p;
	*p = *q;
	*q = t;
}
void exchang(int a[][3])
{
	for (int i = 0; i < 3; i++)
		for (int j = i + 1; j<3; j++)
			jh(&a[i][j], &a[j][i]);
}
void main()
{
	int a[3][3];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			scanf_s("%d", &a[i][j]);
	exchang(a);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			printf("%5d", a[i][j]);
		printf("\n");
	}
}
//3.
#include <stdio.h>
int prime(int n)
{
	int flag = 1;
	for (int i = 2; i <= n / 2 && flag == 1; i++)
		if (n % i == 0) flag = 0;
	return(flag);
}
void main()
{
	int n;
	scanf_s("%d", &n);
	if (prime(n)) printf("Yes.\n"); else printf("No.\n");
}
*/