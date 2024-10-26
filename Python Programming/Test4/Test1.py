def test1():
    def Recursive(n, max_steps):
        if n <= 0:
            return 1
        if n == 1:
            return 1
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n + 1):
            for j in range(1, max_steps + 1):
                if i - j >= 0:
                    dp[i] += dp[i - j]
        return dp[n]

    def Induction(n, max_steps):
        if n <= 0:
            return 1
        if n == 1:
            return 1
        if n < max_steps:
            return Induction(n - 1, max_steps)
        return Induction(n - 1, max_steps) + Induction(n - max_steps, max_steps)
    n = 15
    max_steps = 3
    result_R = Recursive(n, max_steps)
    result_I = Recursive(n, max_steps)
    print("小明上这段楼梯一共有{}种方法(递推)。\n小明上这段楼梯一共有{}种方法(递归)。".format(result_R,result_I))

def test2():

    import random
    import math

    def taken(num):
        if int(math.log2(num+1))==math.log2(num+1) : return False
        return True
    def nimu_game(choice,items):
        while items > 0:
            if choice=='y':
                while True:
                    take=int(input("请你输入你想拿走几个物品："))
                    if ((take<1) or (take>int(items/2))) and (items!=1):
                        print("玩家至少拿走一个并且最多只能拿走一半物品")
                    else:
                        break
                items=items-take
                print("你拿走了", take, "个物品，剩余", items, "个物品。")
                if items==0:
                    print("你输了！")
                    break
                if taken(items) or (items==1):
                    take=items-2**int(math.log2(items))+1
                    items=items-take
                    print("电脑拿走了", take, "个物品，剩余", items, "个物品。")
                    if items==0:
                        print("你赢了！")
                        break
                else:
                    take=random.randint(1,items//2)
                    print("电脑拿走了", take, "个物品，剩余", items, "个物品。")
            if choice=='n':
                if taken(items) or (items==1):
                    take=items-2**int(math.log2(items))+1
                    items=items-take
                    print("电脑拿走了", take, "个物品，剩余", items, "个物品。")
                    if items==0:
                        print("你赢了！")
                        break
                else:
                    take=random.randint(1,items//2)
                    print("电脑拿走了", take, "个物品，剩余", items, "个物品。")
                while True:
                    take=int(input("请你输入你想拿走几个物品："))
                    if ((take<1) or (take>int(items/2)))and (items!=1):
                        print("玩家至少拿走一个并且最多只能拿走一半物品")
                    else:
                        break
                items=items-take
                print("你拿走了", take, "个物品，剩余", items, "个物品。")
                if items==0:
                    print("你输了！")
                    break

    s=input("请你选择先手拿或后手拿\ny代表选择先手\nn代表选择后手\n(y/n):")
    item=int(input("总共的物品数量："))
    nimu_game(s,item)


def test3():

    def game_recursive(n, k):
        if n == 1:
            return 1
        else:
            return (game_recursive(n - 1, k) + k - 1) % n + 1


    def game_loop(n, k):
        people = list(range(1, n + 1))
        index = 0
        while len(people) > 1:
            index = (index + k - 1) % len(people)
            del people[index]
        return people[0]

    n = int(input("请输入初始人数："))
    k = int(input("请输入报数临界值："))
    result_recursive = game_recursive(n, k)
    result_loop = game_loop(n, k)
    print("最后留下的是原来第{}号（递归）。\n最后留下的是原来第{}号（循环）。".format(result_recursive,result_loop))

def test4():

    import random

    def get_random():
        return random.randint(0, 360)

    def get_prize(angle):
        if 0 <= angle < 30:
            return "一等奖"
        elif 30 <= angle < 108:
            return "二等奖"
        else:
            return "三等奖"

    prize_count = {"一等奖": 0, "二等奖": 0, "三等奖": 0}

    for i in range(10000):
        angle = get_random()
        prize = get_prize(angle)
        prize_count[prize] += 1

    print("一等奖中奖次数：", prize_count["一等奖"])
    print("二等奖中奖次数：", prize_count["二等奖"])
    print("三等奖中奖次数：", prize_count["三等奖"])

def test5():
    def hanoi(n, source, target, temp):
        if n == 1:
            print(f"将盘子从 {source} 移动到 {target}")
            return
        hanoi(n - 1, source, temp, target)
        print(f"将盘子从 {source} 移动到 {target}")
        hanoi(n - 1, temp, target, source)

    hanoi(4, 'A', 'C', 'B')

def test6():

    import random

    def catch():
        hole = random.randint(1, 5)
        attempts = 0
        max_attempts = 5

        while attempts < max_attempts:
            player = int(input("请输入一个洞口（1-5）："))
            print("你总共有5次尝试机会")
            attempts += 1

            if player == hole:
                print("恭喜你抓到了狐狸！")
                return
            else:
                print("很遗憾，你没有抓到狐狸。")
                if attempts < max_attempts:
                    path=random.choice([-1,1])
                    if hole==5 : path=-1
                    if hole==1 : path=1
                    hole=hole+path
                    print("狐狸跳到隔壁洞口里了，请继续尝试。")
                else:
                    print("你没有在规定次数内抓到狐狸。游戏失败。")

    catch()

def test7():

    import random

    def game():
        doors = [1, 2, 3]
        car_door = random.choice(doors)
        goat_doors = [door for door in doors if door != car_door]

        chosen_door = random.choice(doors)
        opened_door = [door for door in doors if door != chosen_door and door != goat_doors[0]][0]

        print("你选择了门", chosen_door)
        print("主持人打开了门", opened_door,"后面是一只山羊")

        if input("你想改选门吗？ (y/n): ") == 'y':
            chosen_door = int(input("请输入你想要选择的门号： "))

        if chosen_door == car_door:
            return "你赢得了车！"
        else:
            return "你输了！"

    print(game())


if __name__=='__main__':
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()