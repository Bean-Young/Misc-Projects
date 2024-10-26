#include <iostream>
using namespace std;
//recursive 递归实现阶乘
int f_recursive(int n) 
{
	if (n == 1)
		return 1;
	return n * f_recursive(n - 1);
}
//iterative 迭代实现阶乘
int f_iterative(int n)
{
	int result = 1;
	for (int i = 1; i <= n; i++)
		result *= i;	
	return result;
}

int main()
{
	int x;
	cin >> x;
	cout << f_recursive(x) << endl;
	cout << f_iterative(x) << endl;
	return 0;
}