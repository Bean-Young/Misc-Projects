# -*- coding: utf-8 -*-

def test1():

    import random
    import string
    import codecs

    def randomstring(length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))

    def randomtel():
        return '1' + ''.join(random.choice('0123456789') for i in range(9))

    def randomemail():
        return randomstring(5) + '@' + randomstring(5) + '.com'

    def main(filename):
        with codecs.open(filename, 'w', 'utf-8') as fp:
            fp.write('Name,Sex,Age,TelNO,Address,Email\n')
            # 随机生成200个人的信息
            for i in range(200):
                # 生成信息
                name = randomstring(5)
                sex = random.choice(['男', '女'])
                age = str(random.randint(18, 60))
                tel = randomtel()
                address = randomstring(10)
                email = randomemail()
                line = ','.join([name, sex, age, tel, address, email]) + '\n'
                fp.write(line)

    main('people_info.txt')

def test2():
    import re

    text = "行尸走肉、金蝉脱壳、百里挑一、金玉满堂、背水一战、霸王别姬、天上人间、不吐不快、海阔天空、情非得已、满腹经纶、兵临城下、春暖花开、插翅难逃、黄道吉日、天下无双、偷天换日、两小无猜、卧虎藏龙、珠光宝气、簪缨世族、花花公子、绘声绘影、国色天香、相亲相爱、八仙过海、金玉良缘、掌上明珠、皆大欢喜、浩浩荡荡、平平安安、秀秀气气、斯斯文文、高高兴兴"
    pattern = r'((.)\2(.)\3)'
    result=[]
    for text in text.split('、'):
        if re.match(pattern, text) : result.append(text)
    print(result)

def test3():

    import re

    def remove(s):
        return ''.join(re.findall(r'(\w)(?!.*\1)', s[::-1]))[::-1]

    s = input()
    result = remove(s)
    print(result)


if __name__=='__main__':
    test1()
    test2()
    test3()