
#1
print("Hello World")
#2
print(3+5)
#3
x=3;y=5
z=x+y
print(z)
#4
s1,s2,s3=input("please input your scores.\n").split(",")
s=(int(s1)+int(s2)+int(s3))/3
print("Average Score=",s)
#tset1
print("My name is YangYuezhe.")
print("My name is YangYuezhe, and my age is 19.")
#test2
name,age=input("Your name and age\n").split()
print(name)
print(int(age)+3)
#test3
import math
r=int(input("please input a radius.\n"))
s=math.pi*r*r
c=2*math.pi*r
print("Square=",s)
print("Circumference=",c)
#test4
import numpy as np
t = np.random.random() * np.pi*2
x = np.cos(t)
y = np.sin(t)
#注释见小结
print([x,y])
"""
import numpy as np
t = np.random.random() *2
while (t==0) or (t==0.5) or (t==1.5):
    t = np.random.random() * 2
t=t*np.pi
x = np.cos(t)
y = np.sin(t)
print([x,y])
"""