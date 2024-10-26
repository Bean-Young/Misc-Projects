#include <iostream>
#include <fstream>
#include <vector>
#include <string>
using namespace std;

struct HuffManCode {
    string ch;
    string code;
};

vector<HuffManCode> readHuffmanCode() {
    vector<HuffManCode> codes;
    ifstream infile("E:/C/Project 03/HCode.data");
    string ch, code;
    while (infile >> ch >> code) {
        codes.push_back({ ch, code });
    }
    // 特殊处理空格字符
    codes.push_back({ " ", "000" });
    infile.close();
    return codes;
}
int main() {
    auto codes = readHuffmanCode();

    ifstream codefile("E:/C/Project 03/CodeFile.data");
    string encodedStr;
    getline(codefile, encodedStr);
    cout << "读取CodeFile.data: " << encodedStr << endl;

    ofstream textfile("E:/C/Project 03/Textfile.txt");
    cout << "译码结果为:";

    string currentCode;
    for (char bit : encodedStr) {
        currentCode += bit;
        for (const auto& huffmanCode : codes) {
            if (currentCode == huffmanCode.code) {
                textfile << huffmanCode.ch;
                cout << huffmanCode.ch;
                currentCode = ""; // 清空当前编码，准备解码下一个字符
                break;
            }
        }
    }

    cout << "\n译码结果已保存在Textfile.txt中！" << endl;

    codefile.close();
    textfile.close();

    return 0;
}
