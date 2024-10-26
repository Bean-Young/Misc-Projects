#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
char* fun(char* s)
{
    int i, j, k, n; char* p, * t;
    n = strlen(s) + 1;
    t = (char*)malloc(n * sizeof(char));
    p = (char*)malloc(n * sizeof(char));
    j = 0;  k = 0;
    for ( i = 0; i < n; i++)
    { if (((s[i] >= 'a') && (s[i] <= 'z')) || ((s[i] >= 'A') && (s[i] <= 'Z')))
      {
        t[j] = s[i];  j++;
    }
else { p[k] = s[i]; k++; }
    }
    for (i = 0; i < j + k; i++)
        t[j + i] = p[i];
    t[j + k] ='\0';
    return t;
}
void main()
{
    char s[80];
    printf("please input : ");
    scanf("%s", s);
    printf("\nThe result is : %s\n", fun(s));
}
/*
//12.
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#define LEN sizeof(struct student)
int n = 0;
struct student {
    int age;
    long num;
    char name[20];
    unsigned char sex;
    struct student* next;};
struct student* creat()
{                                                
    struct student* head, * p1, * p2;
    head = 0;
    p1 = p2 = (struct student*)malloc(LEN);
    printf("请输入年龄，学号，姓名，性别：\n");
    scanf("%d %ld %s %c", &p1->age, &p1->num, p1->name, &p1->sex);
    while (p1->num != 0) {
        n++;
        if (n == 1) head = p1;
        else p2->next = p1;
        p2 = p1;
        p1 = (struct student*)malloc(LEN);
        printf("请输入年龄，学号，姓名，性别：\n");
        scanf("%d %ld %s %c", &p1->age, &p1->num, p1->name, &p1->sex);

    }
    p2->next = 0;
    return head;
}
struct student* deldate(struct student* p) 
{                             
    int dnum;
    printf("请输入要删除的学生的年龄:\n");
    scanf("%d", &dnum);
    struct student* head, * p1, * p2;
    head = p1 = p2 = p;
    while (p1 != 0)
    {
        if (p1->age == dnum)
        {
            if (head == p1) head = p1->next;
            else p2->next = p1->next;
            p1 = p1->next;
        }
        else 
        {
            p2 = p1;
            p1 = p1->next;
        }
    }
    return head;
}
void print(struct student* p1)
{                                    
    while (p1 != 0)
    {
        printf("%d %ld %s %c\n", p1->age, p1->num, p1->name, p1->sex);
        p1 = p1->next;
    }
}
void main()
{
    struct student* head = creat();
    struct student* head1 = deldate(head);
    print(head1);
}
/*
//6.
#include <stdio.h>
struct person {
    int num;
    struct person* next;
}per[13];
void main()
{
    struct person* p1 = &per[0];
    struct person* p2 = &per[0];
    int i, j = 1;

    for (i = 0; i < 13; i++) 
    {
        per[i].num = i;
        if (i != 12) per[i].next = &per[i + 1];
        else per[i].next = &per[0];
    }

    while (i > 1)
    {
        if (j == 3)
        {
            j = 1;
            p2->next = p1->next;
            p1 = p1->next;
            i--;

        }
        else {
            p2 = p1;
            p1 = p1->next;
            j++;
        }
    }
    printf("最后剩下第%d位 ", p2->num + 1);
}
//5.
#include<stdio.h>
struct student {
    int num;
    char name[8];
    int score[3];
}s[10];
void input(struct student s[])
{
    int i, j;
    for (i = 0; i < 10; i++)
    {
        printf("请输入第%d学生的学号:\n", i + 1);
        scanf_s("%d", &s[i].num);
        printf("请输入第%d学生的姓名：\n", i + 1);
        scanf_s("%s", s[i].name,8);
        printf("请输入第%d学生的三个成绩：\n", i + 1);
        for (j = 0; j < 3; j++)
            scanf_s("%d", &s[i].score[j]);
    }
}
void print(struct student s[])
{
    int i, j;
    float sum[3] = { 0 };
    printf("3门课的总平均成绩：\n");
    for (j = 0; j < 3; j++)
    {
        for (i = 0; i < 5; i++)
            sum[j] += s[i].score[j];
        printf("%8.2f\n", sum[j] / 4);
    }
    float max = s[0].score[0];
    int m=0, n=0;
    for (i = 0; i < 5; i++)
    {
        for (j = 0; j < 3; j++)
            if (max < s[i].score[j])
            {
                max = s[i].score[j];
                m = i; n = j;
            }
    }
    float aver = 0;
    for (j = 0; j < 3; j++)
        aver += s[m].score[j];
    printf("最高分为:%2.2f\n", max);
    printf("最高分的学生的数据为：\n");
    printf("%4d %4s %4.2d %4.2d %4.2d %4.2f\n", s[m].num, s[m].name, s[m].score[0], s[m].score[1], s[m].score[2], aver / 3);
}
void main()
{
    printf("请输入学号，姓名，三个成绩：\n");
    input(s);
    print(s);
}
//4.
#include<stdio.h>
struct student {
    int num;
    char name[8];
    int score[3];
}s[5];
void input(struct student s[])
{
    int i, j;
    for (i = 0; i < 5; i++)
    {
        printf("请输入第%d学生的学号:\n", i + 1);
        scanf_s("%d", &s[i].num);
        printf("请输入第%d学生的姓名：\n", i + 1);
        scanf_s("%s", s[i].name,8);
        printf("请输入第%d学生的三个成绩：\n", i + 1);
        for (j = 0; j < 3; j++)
            scanf_s("%d", &s[i].score[j]);
    }
}
void print(struct student s[])
{
    int i, j;
    for (i = 0; i < 5; i++)
    {
        printf("%5d%10s\t", s[i].num, s[i].name);
        for (j = 0; j < 3; j++)
            printf("%d\t", s[i].score[j]);
        printf("\n");
    }
}
void main()
{
    printf("请输入学号，姓名，三个成绩：\n");
    input(s);
    print(s);
}
//3.
#include<stdio.h>
struct student {
    int num;
    char name[8] = { '\0' };
    int score[3];
}s[5];
void print(struct student s[])
{
    int i, j;
    for (i = 0; i < 5; i++)
    {
        printf("%5d%10s\t", s[i].num, s[i].name);
        for (j = 0; j < 3; j++)
            printf("%d\t", s[i].score[j]);
        printf("\n");
    }
}
void main()
{
    printf("请输入学号，姓名，三个成绩：\n");
    int i, j;
    for (i = 0; i < 5; i++)
    {
        printf("请输入第%d学生的学号:\n", i + 1);
        scanf_s("%d", &s[i].num);
        printf("请输入第%d学生的姓名：\n", i + 1);
        scanf_s("%s",s[i].name,8);
        printf("请输入第%d学生的三个成绩：\n", i + 1);
        for (j = 0; j < 3; j++)
            scanf_s("%d", &s[i].score[j]);
    }
    print(s);
}
//1(2).
#include<stdio.h>
int data(struct days d);
struct days {
    int year;
    int month;
    int day;
}d;
int main()
{
    int n;
    printf("请输入年，月，日:\n");
    scanf_s("%d%d%d", &d.year, &d.month, &d.day);
    n = data(d);
    printf("%d年%d月%d日是%d年的第%d天\n", d.year, d.month, d.day, d.year, n);
}
int data(struct days d)
{
    int smonth[13] = { 0,31,28,31,30,31,30,31,31,30,31,30,31 };
    int i, sum = 0;
    for (i = 0; i < d.month; i++)
        sum += smonth[i];
    sum += d.day;
    if (d.month > 2 && ((d.year % 4 == 0 && d.year % 100 != 0) || d.year % 400 == 0))
        sum++;
    return sum;
}
*/