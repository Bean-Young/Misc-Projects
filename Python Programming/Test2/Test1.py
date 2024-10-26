from functools import reduce
def test1():
    x=input("Please enter a natural number\n")
    print("The sum of all the numbers:",sum(map(int,x)))
def test2():
    setA = eval(input("Please enter a set\n"))
    setB = eval(input("Please enter a set\n"))
    print("Intersection:", setA | setB)
    print("Union:", setA & setB)
    print("Difference:", setA - setB)
def test3():
    x = int(input("Please enter a natural number\n"))
    print("Binary:", bin(x))
    print("Decimal:", oct(x))
    print("Hexadecimal:", hex(x))
def test4():
    lis=eval(input("Please enter a list containing several integers\n"))
    lis=list(filter(lambda x: x % 2 == 0,lis))
    print("List only with even number:",lis)
def test5():
    lisA=eval(input("Please enter a listA\n"))
    lisB=eval(input("Please enter a listB\n"))
    dictory = dict(zip(lisA,lisB))
    print("Dictory:",dictory)
def test6():
    lis= eval(input("Please enter a list containing several integers\n"))
    lis.sort(key=None, reverse=True)
    print("Descend order:",lis)
def test7():
    lis=eval(input("Please enter a list containing several integers\n"))
    result=reduce(lambda x,y:x*y,lis)
    print("The result of multiplying all integers in the list:",result)
def test8():
    pointA = eval(input("Enter coordinate pointA\n"))
    pointB = eval(input("Enter coordinate pointB\n"))
    distance = map(lambda x, y: abs(x - y), pointA, pointB)
    print("Manhattan distance between two points:", sum(distance))
def test9():
    sets=eval(input("Please enter a list containing several sets\n"))
    union_set=reduce(lambda x,y:x.union(y),sets)
    print("Union of the sets:",union_set)
def test10():
    a1 = int(input("Please enter the first item:"))
    q = int(input("Please enter the common ratio:"))
    n = int(input("Please enter a natural number n:"))
    sum = int(a1* (1 - q ** n) / (1 - q))
    print("The sum of the first n terms in a proportional sequence:",sum)
def test11():
    str = input("Please enter a string of characters\n")
    dic= {}
    for i in str:
        dic[i] = str.count(i)
    v = max(dic.values())
    for key, value in dic.items():
        if (value == v):
            print("The character with the highest number of occurrences and its number of occurrences:",'"'+key+'"', v)
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
    test10()
    test11()
