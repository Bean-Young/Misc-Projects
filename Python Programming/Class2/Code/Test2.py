def is_prime(n):
    if n < 2:
        return False
    elif n == 2:
        return True
    else:
        a = int(n ** (1 / 2) + 1)
        for i in range(2, a + 1):
            if n % i == 0:
                return False
        return True
def test1():
    lst=[i for i in range(1,11) if is_prime(i)]
    print(lst)
def test2():
    lst=[i for i in range(1,101) if i%2==0]
    print(lst)
def test3():
    lst=[[2,-1,3],[-2,5,1]]
    print(lst)
def test4():
    scores = [78, 69, 53, 97, 88, 31, 74, 92]
    pa=sum(map(lambda score:score//60,scores))
    print('{:.2%}'.format(pa/len(scores)))
def test5():
    global x
    x=input("Please enter scores for ten students\n").split()
    x=list(map(int,x))
    print(x)
def test6():
    global x
    x=list(filter(lambda score:0<=score<=100,x))
    print(x)
def test7():
    str=input("Please enter a string\n")
    str=list(str)
    counta=str.count('a')+str.count('A')
    counte=str.count('e')+str.count('E')
    counti=str.count('i')+str.count('I')
    counto=str.count('o')+str.count('O')
    countu=str.count('u')+str.count('U')
    print("Number of occurrences of letter a:",counta)
    print("Number of occurrences of letter e:",counte)
    print("Number of occurrences of letter i:",counti)
    print("Number of occurrences of letter o:",counto)
    print("Number of occurrences of letter u:",countu)
def test8():
    scores=eval(input("Please enter scores for students\n"))
    scores=list(filter(lambda score:score>=60,scores))
    print(scores)
def test9():
    str = input("Please enter a string of characters\n")
    dic= {}
    for i in str:
        dic[i] = str.count(i)
    v = max(dic.values())
    for key, value in dic.items():
        if (value == v):
            print("The character with the highest number of occurrences and its number of occurrences:",key, v)
if __name__ == '__main__':
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    test8()
    test9()