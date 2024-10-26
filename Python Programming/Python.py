
#12.03 
Python 可以同时对多个变量定义并赋值
a,b=0,1
print(a)
print(b)
输入输出格式
name=input('')　输入
print(value1,value2,...,sep=' ',end='\n')　输出
sep默认为空格 end默认为换行
字符串　三引号多行字符
转义字符　\\ \' \" \000 \oyy 八进制　\xyy　十六进制
\other 输出其他字符
string.join　连接字符串
string.split 分割字符串
len(string)
string.isalnum() 是否包含0-9 a-z A-Z
string.title() 首字母大写
0o[OO] 八进制
0x[OX]　十六进制
0b[OB]　二进制
// 整除
% 取余
** 乘方
type()　查询变量的数据类型
complex a+bj a,b均为浮点型
比较运算符 (== !=)　赋值运算符 (**= //=)　和Ｃ相同　
and or not 逻辑表达式
in (not in) 成员运算符　返回布尔值
# if (a in list):
身份运算符　
变量有三个属性　name id value
通过　id() 函数　查看当前变量id
a is b is 身份运算符　返回布尔值　比较ｉｄ
与　== 的区别　== 比较　value
a=b=c=1 a,b,c id value　均相同
a,b,c=1,1,1 a,b,c value 相同　id　不同
if condition_1:
    statement_block_1
elif condition_2:
    statement_block_2
else:
    statement_block_3
for iterating_var in sequence:
    statements(s)
range([start,]stop[,step])　到stop-1
for 
else:
else 中语句会在循环正常执行完的情况下执行
while condition:
    statements
else:
    statements
crtl+c 退出死循环
continue 结束此次循环
break　结束此个循环
pass 空语句
numbers=list(range(1,4))
[1,2,3]
numbers[1:2]
[2,3]
############
print()
语言设置
变量命名规则相同 建议小写
调试与错误
字符串 string
方法:name.title() 首字母 name.upper() 大写name.lower() 小写 注意括号
多行字符串可以用 三单引或三双引
字符串+
\t 空格符号 制表符 \n 换行符 可以在变量中使用吗？ 可以 s="\t\n"
方法 name.rstrip() 除去末尾空格 name=name.rstrip() lstrip() 开头空格 strip() 两头空格
字符串注意双引号与单引号区别
整数
**表乘方
浮点数（小数）
Python最大区别 变量不需命名 直接开 设置类型 注意类型很重要
函数str() 转字符串
浮点数与整数 3/2整数会显示1 3.0/2 即为1.5
# 注释 单行 
'''三单引或三双引 多行注释'''
关于数据类型 https://m.php.cn/article/454931.html
type()函数 查看当前数据类型
Unicode 编码类型 在定义字符前加u或U 如 name=U'我'
有关数据类型的转换https://blog.csdn.net/cheuklam/article/details/120555191
以及相关函数
列表(数组) 字符串类
Python中采用[]表示列表 如 names=[1,2,3]
同时注意索引从0开始 names[0]=1
同时 从后索引 names[-1]=3 name[-2]=2
names.append(1) 末尾添加1 [1,2,3,1]
names.insert(2,33) 与2索引位置插入数字33 [1,2,33,3,1]
del names[1] [1,33,3,1]
pop 函数 弹出 末尾开始 name=names.pop() names [1,33,3] name 1
pop中可以添加位置 如name=names.pop(0) names [33,3,1] name 1
remove 函数 不知道位置但知道删除内容 names.remove(1) [33,3] 如果列表中有重复呢？删除第一个相应的元素 如果没有 将报错
sort 方法 names.sort() 顺序 适用于数字
names.sort(reserve=True) True T必须大写 倒序
sorted 函数 sorted(names) 临时排序
reverse() 方法 永久性 反序 最后一位变第一位
len() 函数 确实列表长度
for循环格式 for name in names: <:>不可遗漏
缩进 这点是Python中较重要的一点
Python通过缩进将格式模块化
数字列表
range()函数生成数字 注意 range(1,4)是 1,2,3
计算平方  示例
squares=[]
for value in range(1,11):
    square=value**2
    squares.append(square)
print(squares)
min()  max() sum() 函数对应数字列表的处理
列表解析  squares=[ value**2 for value range(1,11)]  一行处理刚刚平方问题 没有冒号 简化程序
切片 对列表部分元素的处理
print(names[0:3]) 打印names列表的一二三个元素
0开始可以省略 names[:3] 到末尾也可以省略 names[0:]
print(names[-3:]) 打印最后三项for name in names[0:2]: 部分的循环
num=names[:] 复制整个列表
num=names 与 num=names[:] 的区别 第一种是副本 之后会随names的改动而改动 第二种无关
不可变的列表 为元组
num=(2,3) 定义元组 不可改变 但是可以重新定义
python 格式设置 PEP 8 建议每级缩进为四格 制表符转换为空格 单行不超过79字符
