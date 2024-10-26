#include <vector>
#include <iostream>

int main() {
    std::vector<int> vec; // 创建一个空的vector

    // 添加元素
    vec.push_back(10);
    vec.push_back(20);

    // 访问元素
    std::cout << "第一个元素: " << vec[0] << std::endl;

    // 使用迭代器访问元素
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    // 修改元素
    vec[0] = 5;

    // 删除元素
    vec.pop_back(); // 删除最后一个元素

    // 获取vector的大小
    std::cout << "Vector size: " << vec.size() << std::endl;
    getchar();
    return 0;
}
