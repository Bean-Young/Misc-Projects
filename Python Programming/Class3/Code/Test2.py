def test1():
    import string

    def password_strength(password):
        has_digit=0
        has_lower=0
        has_upper=0
        has_punctuation=0

        for char in password:
            if char.isdigit():
                has_digit = 1
            elif char.islower():
                has_lower = 1
            elif char.isupper():
                has_upper = 1
            elif char in string.punctuation:
                has_punctuation = 1

        ans=has_digit+has_punctuation+has_lower+has_upper
        if ans==4:
            return "强密码"
        elif ans==3:
            return "中育密码"
        elif ans==2:
            return "中弱密码"
        else:
            return "弱密码"

    password = input("请输入密码：")
    print("密码强度为：", password_strength(password))

def test2():
    def caesar_encrypt(plaintext, key):
        ciphertext = ""
        for char in plaintext:
            if char.isalpha():
                shift = key % 26
                if char.islower():
                    ciphertext += chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
                else:
                    ciphertext += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            else:
                ciphertext += char
        return ciphertext

    plaintext = input("请输入待加密的明文：")
    key = int(input("请输入凯撒加密算法的密钥："))
    ciphertext = caesar_encrypt(plaintext, key)
    print("加密后的结果为：", ciphertext)

def test3():
    def typing_practice(origin, userInput):
        if len(userInput) > len(origin):
            return "错误：用户输入内容长度超过原始内容"

        correct_count = 0
        for i in range(len(userInput)):
            if origin[i] == userInput[i]:
                correct_count += 1

        score = correct_count / len(origin) * 100
        return f"成绩：{score:.2f}%"

    origin = "Positron Emission Tomography is a nuclear medicine imaging technique used to observe specific aspects of biological activity in human or animal bodies."
    userInput=input()
    print(typing_practice(origin, userInput))

def test4():
    def classify_email(content):
        symbols = ['[', ']', '*', '-', '/']
        total_count = len(content)
        symbol_count = sum([content.count(symbol) for symbol in symbols])
        ratio = symbol_count / total_count
        return(ratio)

    ratio=classify_email(input())
    if ratio > 0.2:
        print("垃圾邮件")
    else:
        print("正常邮件")

if __name__=='__main__':
    test1()
    test2()
    test3()
    test4()