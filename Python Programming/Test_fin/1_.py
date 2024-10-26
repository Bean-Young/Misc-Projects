str1=input()
str2=input()
lenth_max=-1
i_i=0
i_j=0
for i in range(0,len(str1)):
    for j in range(0,len(str2)):
        if str1[i]==str2[j]:
            index_i=i
            index_j=j
            lenth=0
            while (index_i<len(str1)) and (index_j<len(str2)) and (str1[index_i]==str2[index_j]) :
                index_i=index_i+1
                index_j=index_j+1
                lenth=lenth+1
            if lenth>lenth_max:
                lenth_max=lenth
                i_i=i
                i_j=index_i-1
print(str1[i_i:i_j+1])

