#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <cctype> 
using namespace std;

int main()
{
    cout << "Start " << endl;
    ifstream inFile("E:/C/Project 03/Words.txt");
    if (!inFile)
    {
        cout << "File not found!" << endl;
        return 0;
    }

    int letterCount[27] = { 0 };
    char c;
    while (inFile.get(c))
    {
        if (isalpha(c))
        {
            c = tolower(c);
            letterCount[c - 'a']++; 
        }
        else if (c == ' ')
        {
            letterCount[26]++; 
        }
    }

    ofstream outFile("DataFile.data");
    if (!outFile)
    {
        cout << "Failed to create output file!" << endl;
        return 0;
    }


    for (int i = 0; i < 26; i++)
    {
        outFile << (char)('a' + i) << " " << letterCount[i] << endl;
    }

    outFile << "Space " << letterCount[26] << endl;

    cout << "Counting completed! Result saved in DataFile.data." << endl;

    inFile.close();
    outFile.close();

    return 0;
}
