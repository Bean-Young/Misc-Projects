#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

// 员工类
class Employee {
private:
    string phone;
    string name;
    string address;

public:
    Employee(const string& phone, const string& name, const string& address) : phone(phone), name(name), address(address) {}

    const string& getPhone() const { return phone; }
    const string& getName() const { return name; }
    const string& getAddress() const { return address; }
};

// 哈希表类
class HashTable {
private:
    static const int SIZE = 10; // 哈希表大小
    vector<Employee*> table;

    // 哈希函数
    unsigned int hash(const string& key) const {
        unsigned int hashValue = 0;
        for (char c : key) {
            hashValue = (hashValue << 5) + c;
        }
        return hashValue % SIZE;
    }

public:
    HashTable() {
        table.resize(SIZE, nullptr);
    }

    // 插入员工信息
    void insert(const Employee& emp) {
        int index = hash(emp.getPhone());
        while (table[index] != nullptr) {
            index = (index + 1) % SIZE;
        }
        table[index] = new Employee(emp);
    }

    // 根据电话号码查找员工信息
    Employee* search(const string& phone) const {
        int index = hash(phone);
        while (table[index] != nullptr && table[index]->getPhone() != phone) {
            index = (index + 1) % SIZE;
        }
        if (table[index] != nullptr && table[index]->getPhone() == phone) {
            return table[index];
        }
        return nullptr;
    }

    // 保存通讯录信息到文件
    void saveToFile(const string& filename) const {
        ofstream file(filename);
        if (!file.is_open()) {
            cout << "Error opening file." << endl;
            return;
        }
        for (Employee* emp : table) {
            if (emp != nullptr) {
                file << emp->getPhone() << "," << emp->getName() << "," << emp->getAddress() << endl;
            }
        }
        file.close();
    }

    // 析构函数释放内存
    ~HashTable() {
        for (Employee* emp : table) {
            delete emp;
        }
    }
};

int main() {
    HashTable ht;

    // 从键盘输入员工信息
    string phone, name, address;
    int choice;
    do {
        cout << "Menu:\n1. Add employee information\n2. Search employee information\n3. Save contacts to file\n4. Exit\nEnter your choice: ";
        cin >> choice;

        switch (choice) {
        case 1:
            cout << "Enter employee information (phone, name, address): ";
            cin >> phone >> name >> address;
            ht.insert(Employee(phone, name, address));
            // 清空输入缓冲区中的换行符
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            break;
        case 2: {
            cout << "Enter a phone number to search: ";
            cin >> phone;
            // 清空输入缓冲区中的换行符
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            Employee* emp = ht.search(phone);
            if (emp != nullptr) {
                cout << "Name: " << emp->getName() << endl;
                cout << "Address: " << emp->getAddress() << endl;
            }
            else {
                cout << "Employee not found." << endl;
            }
            break;
        }
        case 3: {
            cout << "Enter filename to save: ";
            string filename;
            cin >> filename;
            ht.saveToFile(filename);
            break;
        }
        case 4:
            cout << "Exiting program." << endl;
            break;
        default:
            cout << "Invalid choice. Please try again." << endl;
        }
    } while (choice != 4);

    return 0;
}
