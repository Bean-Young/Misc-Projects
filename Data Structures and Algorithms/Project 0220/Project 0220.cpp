#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
struct Student {
	int number;
	char name[10];
	float score[2];
	float score_ave;
};
#define N 5
void scan(struct Student stu[])
{
	FILE* fp;
	fp=fopen("file.txt", "w");
	for (int i = 0; i < N; i++)
	{
		scanf("%d %s %f %f", &stu[i].number, stu[i].name, &stu[i].score[0], &stu[i].score[1]);
		fprintf(fp,"%d %s %5.1f %5.1f\n", stu[i].number, stu[i].name, stu[i].score[0], stu[i].score[1]);
	}
	fclose(fp);
}
void average(struct Student stu[])
{
	float class1=0, class2=0;
	printf("学号　　　　名字　　成绩1　　成绩2　　平均成绩\n");
	for (int i = 0; i < N; i++)
	{
		stu[i].score_ave = (stu[i].score[0] + stu[i].score[1]) / 2.0;
		class1 += stu[i].score[0];
		class2 += stu[i].score[1];
		printf("%5d %10s   %5.1f %5.1f  %5.1f\n", stu[i].number, stu[i].name, stu[i].score[0], stu[i].score[1], stu[i].score_ave);
	}
	printf("第一门课平均成绩:%5.1f  第二门课平均成绩:%5.1f\n",class1/N,class2/N);
}
void print_max(struct Student stu[])
{
	int m = 0;
	for (int i = 1; i < N; i++)
		m = stu[i].score_ave > stu[m].score_ave ? i : m;
	printf("成绩最高的\n");
	printf("学号：%d\t姓名：%s\t成绩：%5.1f,%5.1f\t平均成绩：%5.1f\n", stu[m].number, stu[m].name, stu[m].score[0], stu[m].score[1], stu[m].score_ave);
}
int main()
{
	struct Student stu[N], * p = stu;
	scan(p);
	average(p);
	print_max(p);
	return 0;
}
