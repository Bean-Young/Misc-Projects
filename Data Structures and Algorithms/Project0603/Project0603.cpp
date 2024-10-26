#include <iostream>
#include <cstring>
#pragma warning (disable:4996)
using namespace std;
#define MAXLEN 255
#define OK 1
#define ERROR 0
#define OVERFLOW -2
typedef int Status;
typedef struct
{
	char ch[MAXLEN + 1];
	int length;
}SString;
int nextp[MAXLEN + 1] = { 0 };
int nextval[MAXLEN + 1] = { 0 };
Status StringAssign(SString& S, const char* str)
{
	strcpy(S.ch + 1, str);
	S.length = strlen(str);
	return OK;
}
int Index_BF(SString S, SString T, int pos)
{
	int i = pos; int j = 1;
	while (i <= S.length && j <= T.length)
	{
		if (S.ch[i] == T.ch[j])
		{
			++i;
			++j;
		}
		else
		{
			i = i - j + 2;
			j = 1;
		}
	}
	if (j > T.length)
		return i - T.length;
	else return 0;
}
void next_get(SString T,int next[])
{
	int i = 1; next[1] = 0; int j = 0;
	while (i < T.length)
	{
		if (j == 0 || T.ch[i] == T.ch[j])
		{
			++i;
			++j;
			next[i] = j;
		}
		else
			j = next[j];
	}
}
void nextval_get(SString T, int nextval[])
{
	int i = 1; nextval[1] = 0; int j = 0;
	while (i < T.length)
	{
		if (j == 0 || T.ch[i] == T.ch[j])
		{
			++i;
			++j;
			if (T.ch[i] != T.ch[j])
				nextval[i] = j;
			else nextval[i] = nextval[j];
		}
		else j = nextval[j];
	}
}
int Index_KMP(SString S, SString T, int pos)
{
	int i = pos; int j = 1; 
	while (i <= S.length && j <= T.length)
	{
		if (j == 0 || S.ch[i] == T.ch[j])
		{
			i++;
			j++;
		}
		else
			j = nextp[j];
	}
	if (j > T.length)
		return i - T.length;
	else return 0;
}
void print(int len)
{
	for (int i = 1; i <= len;i++)
		cout << "next[" << i << "]=" << nextp[i] << "\t";
	cout << endl;
	for (int i = 1; i <= len;i++)
		cout << "nextval[" << i << "]=" << nextval[i] << "\t";
	cout << endl;
}
void strmcpy(SString &P, SString &S, int m)
{
	for (int i = 1; i<=P.length; i++)
	{
		S.ch[i]=P.ch[m];
		m++;
	}
	S.length = P.length;
}
void Matching()
{
	SString T, P,S;
	StringAssign(T, "ATCCGTACGAATCGATCCCCCATGC");
	StringAssign(P, "CCCATGATCC");
	StringAssign(S, "");
	for (int i = 1; i <= P.length; i++)
		P.ch[i + P.length] = P.ch[i];
	int i;
	for (i = 1; i <= P.length; i++)
	{
		strmcpy(P, S, i);
		next_get(S, nextp);
		nextval_get(S, nextval);
		if (Index_KMP(T, S, 1))
		{
			cout <<"Yes!" << endl<<Index_KMP(T, S, 1) << endl;
			break;
		}
	}
	if (i == P.length + 1) cout << "No!";
}
int main()
{
	SString Target, Pattern;
	StringAssign(Target, "abcaabbabcabaacbacba");
	StringAssign(Pattern, "abcabaa");
	cout << Index_BF(Target, Pattern, 1) << endl;
	next_get(Pattern,nextp);
	nextval_get(Pattern, nextval);
	print(Pattern.length);
	cout << Index_KMP(Target, Pattern, 1) << endl;
	Matching();
	return 0;
}