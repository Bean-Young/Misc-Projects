str1=input()+"."
lis=['0','1','2','3','4','5','6','7','8','9']
l=[]
num=-1
for i in str1:
    if i in lis:
        if num==-1: num=0
        num=num*10+int(i)
    else:
        if num!=-1:
            l.append(num)
            num=-1
print(len(l))
print(" ".join(list(map(str,l))))
