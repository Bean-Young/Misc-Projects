#include <stdio.h>
#include <stdbool.h>

// 定义全局变量
#define P 5  // 进程数量
#define R 3  // 资源种类数量

int Allocation[P][R] = {
    {0, 1, 0},
    {2, 0, 0},
    {3, 0, 2},
    {2, 1, 1},
    {0, 0, 2}};  // 已分配矩阵

int Max[P][R] = {
    {7, 5, 3},
    {3, 2, 2},
    {9, 0, 2},
    {2, 2, 2},
    {4, 3, 3}};  // 最大需求矩阵

int Need[P][R];  // 需求矩阵，动态计算
int Available[R] = {3, 3, 2};  // 可用资源向量

// 函数声明
void calculateNeed();
bool isSafeState();
bool requestResources(int processID, int request[]);

int main() {
    // 计算需求矩阵
    calculateNeed();

    // (1) T0 时刻的安全性检查
    printf("步骤 (1): T0 时刻的安全性检查\n");
    if (isSafeState()) {
        printf("系统处于安全状态。\n");
    } else {
        printf("系统不处于安全状态。\n");
    }

    // (2) P1 请求资源 Request1(1, 0, 2)
    printf("\n步骤 (2): P1 请求资源 Request1(1, 0, 2)\n");
    int request1[R] = {1, 0, 2};
    if (requestResources(1, request1)) {
        printf("P1 的资源请求被批准。\n");
    } else {
        printf("P1 的资源请求被拒绝。\n");
    }

    // (3) P4 请求资源 Request4(3, 3, 0)
    printf("\n步骤 (3): P4 请求资源 Request4(3, 3, 0)\n");
    int request4[R] = {3, 3, 0};
    if (requestResources(4, request4)) {
        printf("P4 的资源请求被批准。\n");
    } else {
        printf("P4 的资源请求被拒绝。\n");
    }

    // (4) P0 请求资源 Request0(0, 2, 0)
    printf("\n步骤 (4): P0 请求资源 Request0(0, 2, 0)\n");
    int request0[R] = {0, 2, 0};
    if (requestResources(0, request0)) {
        printf("P0 的资源请求被批准。\n");
    } else {
        printf("P0 的资源请求被拒绝。\n");
    }

    // (5) 最终系统安全性检查
    printf("\n步骤 (5): 进行最终的安全性检查\n");
    if (isSafeState()) {
        printf("系统在分配资源后仍处于安全状态。\n");
    } else {
        printf("系统在分配资源后不安全。\n");
    }

    return 0;
}

// 计算每个进程的需求矩阵 Need = Max - Allocation
void calculateNeed() {
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < R; j++) {
            Need[i][j] = Max[i][j] - Allocation[i][j];
        }
    }
}

// 安全性检查算法
bool isSafeState() {
    int Work[R];        // 当前可用资源工作向量
    bool Finish[P] = {false};  // 标记进程是否完成
    int safeSequence[P]; // 保存安全序列

    // 初始化工作向量 Work = Available
    for (int i = 0; i < R; i++) {
        Work[i] = Available[i];
    }

    int count = 0;  // 安全序列计数器

    // 循环检查所有进程
    while (count < P) {
        bool allocated = false;
        for (int i = 0; i < P; i++) {
            if (!Finish[i]) { // 如果进程尚未完成
                bool canAllocate = true;

                // 检查 Need 是否小于等于 Work
                for (int j = 0; j < R; j++) {
                    if (Need[i][j] > Work[j]) {
                        canAllocate = false;
                        break;
                    }
                }

                if (canAllocate) { // 如果可以分配
                    for (int j = 0; j < R; j++) {
                        Work[j] += Allocation[i][j];  // 释放已分配的资源
                    }
                    Finish[i] = true;  // 标记该进程完成
                    safeSequence[count++] = i;  // 添加到安全序列
                    allocated = true;
                }
            }
        }

        if (!allocated) { // 如果没有进程可以分配资源
            return false; // 不安全
        }
    }

    // 输出安全序列
    printf("安全序列: ");
    for (int i = 0; i < P; i++) {
        printf("P%d ", safeSequence[i]);
    }
    printf("\n");

    return true;
}

// 银行家算法中的资源请求部分
bool requestResources(int processID, int request[]) {
    // 检查请求是否小于等于 Need 和 Available
    for (int i = 0; i < R; i++) {
        if (request[i] > Need[processID][i]) {
            printf("进程 P%d 的请求超过最大需求。\n", processID);
            return false;
        }
        if (request[i] > Available[i]) {
            printf("进程 P%d 的请求超过当前可用资源。\n", processID);
            return false;
        }
    }

    // 尝试分配资源
    for (int i = 0; i < R; i++) {
        Available[i] -= request[i];
        Allocation[processID][i] += request[i];
        Need[processID][i] -= request[i];
    }

    // 检查分配后是否安全
    if (isSafeState()) {
        return true;  // 如果安全，分配成功
    } else {
        // 如果不安全，回滚资源分配
        for (int i = 0; i < R; i++) {
            Available[i] += request[i];
            Allocation[processID][i] -= request[i];
            Need[processID][i] += request[i];
        }
        printf("进程 P%d 的请求会导致不安全状态，已回滚。\n", processID);
        return false;
    }
}