def test1():
    import random

    summation = 0
    cnt = int(input("Please enter the number of times you throw darts："))
    for i in range(cnt):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x ** 2 + y ** 2 <= 1:
            summation += 1
    pi1 = summation / cnt * 4
    pi2 = 0
    for k in range(100000):
        pi2 += 16 ** (-k) * (4 / (8 * k + 1) - 2 / (8 * k + 4) - 1 / (8 * k + 5) - 1 / (8 * k + 6))
    print(f'The calculation result of Monte Carlo method：{round(pi1, 8)}')
    print(f'The approximate calculation formula for pi results in：{round(pi2, 8)}')
    print(f'The error：{round(abs(pi1 - pi2), 8)}')

def test2():
    n = int(input("Please enter a natural number greater than 2："))
    numbers = set(range(2, n + 1))
    prime_numbers = set()
    while numbers:
        current = min(numbers)
        prime_numbers.add(current)
        numbers -= set(range(current, n + 1, current))
    print("A set composed of all prime numbers less than n：", prime_numbers)

def test3():
    n = int(input("Please enter a natural number greater than 2："))
    numbers = list(range(2, n + 1))
    prime_numbers = []
    while numbers:
        current = min(numbers)
        prime_numbers.append(current)
        numbers=[i for i in numbers if i not in list(range(current, n + 1, current))]
    print("A list composed of all prime numbers less than n：", prime_numbers)


def test4():
    import random

    def isPrime(n):
        if n <= 1:
            return False
        for i in range(2, n):
            if n % i == 0:
                return False
        return True

    lst = [random.randint(1, 100) for _ in range(50)]
    non_prime_numbers = list(filter(lambda x: not isPrime(x), lst))
    print(non_prime_numbers)

def test5():
    import itertools
    s = list(itertools.combinations(range(10), 4))
    flag = False

    for i in s:
        snum = ''.join(list(map(lambda x: str(x), i)))
        flag = False
        for j in range(7):
            l = sorted(snum)
            min = int(''.join(l))
            max = int(''.join(reversed(l)))
            if (max - min == 6174):
                flag = True
                break
            else:
                snum = str(max - min)
        if (flag == False):
            break

    if (flag):
        print('6174 conjecture is true')
    else:
        print('6174 conjecture is false')

def test6():
    import random

    def guess_number(min_num, max_num, max_tries):
        target = random.randint(min_num, max_num)
        print(f"请在{min_num}到{max_num}之间猜一个数字，你有{max_tries}次机会。")

        for i in range(max_tries):
            guess = int(input("请输入你猜测的数字："))

            if guess == target:
                print("恭喜你猜对了！")
                return
            elif guess < target:
                print("太小了！")
            else:
                print("太大了！")

        print(f"很遗憾，你没有猜对。正确答案是{target}。")

    min_num = 1
    max_num = 10000
    max_tries = 20
    guess_number(min_num, max_num, max_tries)

if __name__=='__main__':
    #test1()
    #test2()
    #test3()
    #test4()
    #test5()
    test6()