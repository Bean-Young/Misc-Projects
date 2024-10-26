
// 第八章

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
//18.
#include <stdio.h>
int main()
{
    int n;
    char *Month0[12]={"January","February","March","April","May","June","July","August","Septemper","October","November","December"};
            //注意要加双引号

    printf("请输入要查询的月份：");
    scanf("%d",&n);

    if(n>=1 && n<=12)
        printf("%s\n",Month0[n-1]);
    else
        printf("输入错误\n");
    return 0;

}
//17.
#include <stdio.h>
#include<string.h>
int main()
{
    int n;
    char str1[1000]={0},*p1,*p2,str2[1000]={0};

    int strcmp1(char *p1,char *p2);

    printf("请输入字符串1：");
    gets(str1);
    printf("请输入字符串2：");
    gets(str2);

    p1=str1;
    p2=str2;

    n=strcmp1(p1,p2);

    printf("返回值是%d\n",n);

}
int strcmp1(char *p1,char *p2)
{
    int i=0,h=0;
    while(*(p1+i)!='\0' && *(p2+i)!='\0')
    {
        if(*(p1+i)!=*(p2+i))
            h=*(p1+i)-*(p2+i);
        i++;
    }

    return h;
}
//16.
#include <stdio.h>
#include<string.h>
int main()
{
    int i,j=0,n=0;
    int len,cnt=0,a[1000]={0},tmp=0;
    char str[1000]={0},*p,temp[1000]={0};

    printf("请输入字符串：");
    gets(str);
    len=strlen(str);
    p=str;

    for(i=0;i<=len;i++)
    {
    
        if(*(p+i)>='0' && *(p+i)<='9')
        {
            cnt++;
            temp[j++]=*(p+i);
        }
        else
        {
            int x=1;
            if(cnt!=0)
            {
                j=cnt;
                for(;cnt>0;cnt--)
                {
                    tmp+=(temp[j-1]-'0')*x;
                    x*=10;
                    j--;
                }
                a[n++]=tmp;
            }
            j=0;
            tmp=0;
            cnt=0;
        }
    }

    printf("整数个数是%d\n",n);
    for(i=0;i<n;i++)
        printf("%d ",a[i]);
    printf("\n");
    return 0;
}
//15.
#include <stdio.h>
#define M 4
#define N 5
float aver[M];
int main()
{
    int i,j;
    int id[M];
    float score[M][N],(*p)[N];        //（*p）要按列取
    void aver1(float (*p)[N]);
    void less60(float (*p)[N],int id[]);
    void best(float (*p)[N],int id[]);

    for(i=0;i<M;i++)
    {
        printf("\n请第%d个学生的学号：",i+1);
        scanf("%d",&id[i]);
        printf("\n请第%d个学生的成绩：",i+1);
        for(j=0;j<N;j++)
            scanf("%f",&score[i][j]);
    }


    p=score;

    aver1(p);
    less60(p,id);
    best(p,id);

    return 0;
}
void aver1(float (*p)[N])
{
    int i;
    float aver,sum=0;
    for(i=0;i<M;i++)
            sum+=*(*(p+i));
    aver=sum/4.0;
    printf("第一门课的平均分是：%.2f\n",aver);
}
void less60(float (*p)[N],int id[M])
{
    int i,j,n;
    float sum;
    for(i=0;i<M;i++)
    {
        sum=0;
        for(j=0;j<N;j++)
            sum+=*(*(p+i)+j);
        aver[i]=sum/5.0;
    }
    for(i=0;i<M;i++)
    {
        int flag=0;
        for(j=0;j<N;j++)
        {
            if(*(*(p+i)+j)<60.0)
                flag++;
        }
        if(flag>2)
        {
            printf("学号为%d的同学有两门以上成绩不及格！其全部成绩及平均成绩如下：\n",id[i]);
            for(n=0;n<N;n++)
                printf("%.2f ",*(*(p+i)+n));
            printf("\n");
            printf("平均成绩：%.2f\n",aver[i]);
        }
    }
}
void best(float (*p)[N],int id[M])
{
    int i,j,flag;
    for(i=0;i<M;i++)
    {
        flag=1;
        if(aver[i]<=90.0)
            flag=0;
        for(j=0;j<N;j++)
            if(*(*(p+i)+j)<=85.0)
                flag=0;
        if(flag==1)
            printf("学号为%d的同学平均成绩在90分以上或全部课程成绩在85分以上\n",id[i]);
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
//13.
#include <stdio.h>
#include<math.h>
int main()
{
    int n;
    double a,b,c,(*p)(double);
    double jifen(double (*)(double),double,double,int);   //第一个参数是float型函数指针，函数的参数是float型
    double fsin(float);
    double fcos(double);
    double fexp(double);
    printf("请输入积分的上限和下限：");
    scanf("%lf %lf",&a,&b);
    printf("请输入矩形法要分成的份数：");
    scanf("%d",&n);

    p=fsin;
    c=jifen(fsin,a,b,n);
    printf("积分fsin的值为%lf：\n",c);

    p=fcos;
    c=jifen(fcos,a,b,n);
    printf("积分fcos的值为%lf：\n",c);
    
    p=fexp;
    c=jifen(fexp,a,b,n);
    printf("积分fexp的值为%lf：\n",c);

    return 0;
}
double jifen(double (*p)(double),double a,double b,int n)
{
    int i;
    double x,h,s;
    h=(a-b)/n;
    x=b;
    s=0;
    for(i=0;i<n;i++)
        {
            x+=h;
            s+=(*p)(x)*h;
        }
    return s;
}

double fsin(double x)
{
    return sin(x);
}

double fcos(double x)
{
    return cos(x);
}

double fexp(double x)
{
    return exp(x);
}
//12.
#include <stdio.h>
#include<string.h>
#define M 10
#define N 1000
int main()
{
    int i;
    char d[M][N];
    char *p[M];
    void sort(char *str[][N]);   //二维数组做形参时，列不能省
    for(i=0;i<M;i++)
        p[i]=d[i];
    printf("请输入%d个等长的字符串：\n",M);
    for(i=0;i<M;i++)
        gets(p[i]);

    printf("\n");

    sort(p);

    printf("排序后的字符串为：\n");
    for(i=0;i<M;i++)
        printf("%s\n",p[i]);
    printf("\n");
    return 0;
}
void sort(char *str[M])
{
    int i,j;
    char *temp;
    for(i=0;i<M;i++)
        for(j=i;j<M;j++)
            if((strcmp(*(str+j),*(str+i)))>0)
            {
                temp=*(str+j);
                *(str+j)=*(str+i);
                *(str+i)=temp;
            }
}
//11.
#include <stdio.h>
#define M 5
#define N 5
int main()
{
    int i,j;
    int d[M][N];
    void chang(int str[][N],int row,int line);   //二维数组做形参时，列不能省
    printf("请输入一个%dx%d数组：\n",M,N);
    for(i=0;i<M;i++)
        for(j=0;j<N;j++)
            scanf("%d",&d[i][j]);

    printf("\n");
    chang(d,M,N);

    printf("转变后的数组为：\n");
    for(i=0;i<M;i++)
    {
        for(j=0;j<N;j++)
            printf("%5d",d[i][j]);        
        printf("\n");
    }                
    return 0;
}
void chang(int str[][N],int row,int line)
{
    int i,j,t,k,tmp,min_id[4],min_tmp;
    int *p;
    int *pmax,*pmin;

    p=&str[0][0];
    pmax=p;
    pmin=p;

    for(i=0;i<row;i++)
        for(j=0;j<line;j++)
        if(*pmax<*(p+line*i+j))
            pmax=p+line*i+j;

    tmp=*(p+line*row/2);    //行列相乘的总数除以2加一即为中心的位置，注意数组下标是从0开始的
    *(p+line*row/2)=*pmax;
    *pmax=tmp;

    for(i=0;i<4;i++)
    {
        min_tmp=str[row*line-1];
        for(j=0;j<row*line;j++)
        {
            k=0;
            for(;k<i;k++)
                if(j==min_id[k])        //如果某个下标值已经是最小下标，则不再比较
                    break;
            if(k!=i)                    //k!=i说明执行break了，即这时j值是最小下标，结束本次循环
                continue;
            if(min_tmp>str[j])
            {
                min_tmp=str[j];
                min_id[i]=j;
            }
        }
    }
        tmp=*(p);
        *(p)=*(p+min_id[0]);
        *(p+min_id[0])=tmp;

        tmp=*(p+line-1);
        *(p+line-1)=*(p+min_id[1]);
        *(p+min_id[1])=tmp;

        tmp=*(p+line*(row-1));
        *(p+line*(row-1))=*(p+min_id[2]);
        *(p+min_id[2])=tmp;

        tmp=*(p+line*row-1);
        *(p+line*row-1)=*(p+min_id[3]);
        *(p+min_id[3])=tmp;
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
//8.
#include <stdio.h>
int main()
{
    char b[240]={0};

    int a=0,A=0,num=0,space=0,other=0;
    char *p;
    p=b;

    printf("请输入字符串：\n");
    gets(b);

    while(*p!='\0')
    {
        if(*p>='a' && *p<='z')
            a++;
        else if(*p>='A' && *p<='Z')
            A++;
        else if(*p>='0' && *p<='9')
            num++;
        else if(*p==' ')
            space++;
        else
            other++;
        p++;
    }
    printf("小写字母个数：%d\n大写字母个数：%d\n数字个数：%d\n空格个数：%d\n其它字符个数：%d\n",a,A,num,space,other);

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
//4.
#include <stdio.h>
int main()
{
    int i,n,m;
    int num[100]={0};
    void move(int *,int,int);
    printf("请输入数字的个数：\n");
    scanf("%d",&n);
    printf("请依次输入数字：\n");
    for(i=0;i<n;i++)
        scanf("%d",&num[i]);
    printf("请输入要后移的距离：\n");
    scanf("%d",&m);

    move(num,n,m);
    for(i=0;i<n;i++)
        printf("%d",num[i]);
    printf("\n");
    return 0;
}
void move(int *arry[],int n,int m)
{
    int i,j,end=n-m,temp;            //end为倒数第m个数的位置
    int *p;
    for(i=0;i<m;i++)
    {
        p=arry+end+i;                //p指向数组第m+i的地址，即后m个数要前移的数的地址
        temp=*p;                    //用temp记下要前移的数
        for(j=i+end;j>i;j--)        //把前n-m个数依次后移一个单位
        {
            *p=*(p-1);
            p--;
        }
        *(arry+i)=temp;                //把后m个数移到数组前m的位置
    }
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

// 第七章

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
//10. 利用指针使函数返回多个值
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

// Test

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