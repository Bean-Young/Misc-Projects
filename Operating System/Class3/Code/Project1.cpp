#include <stdio.h>
#include <stdbool.h>

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

int Need[P][R];  // 需求矩阵
int Available[R] = {3, 3, 2};  // 可用资源向量

// 函数声明
void calculateNeed();
bool isSafeState(const char* phase);  // 安全性检查
bool requestResources(int processID, int request[], const char* phase);  // 资源请求
void printSystemState(const char* message);  // 输出当前系统状态

int main() {
    // 初始化需求矩阵
    calculateNeed();

    // (1) 检查系统初始安全性
    printf("步骤 (1): 检查系统初始安全性\n");
    isSafeState("(1) 初始安全性检查");

    // (2) P1 请求资源
    printf("\n步骤 (2): P1 请求资源 Request1(1, 0, 2)\n");
    int request1[R] = {1, 0, 2};
    requestResources(1, request1, "(2) P1 请求资源");

    // (3) P4 请求资源
    printf("\n步骤 (3): P4 请求资源 Request4(3, 3, 0)\n");
    int request4[R] = {3, 3, 0};
    requestResources(4, request4, "(3) P4 请求资源");

    // (4) P0 请求资源
    printf("\n步骤 (4): P0 请求资源 Request0(0, 2, 0)\n");
    int request0[R] = {0, 2, 0};
    requestResources(0, request0, "(4) P0 请求资源");

    // (5) 最终安全性检查
    printf("\n步骤 (5): 进行最终安全性检查\n");
    isSafeState("(5) 最终安全性检查");

    return 0;
}

// 计算需求矩阵 Need = Max - Allocation
void calculateNeed() {
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < R; j++) {
            Need[i][j] = Max[i][j] - Allocation[i][j];
        }
    }
}

// 输出当前系统状态
void printSystemState(const char* message) {
    printf("[%s] 系统状态:\n", message);
    printf("Available: ");
    for (int i = 0; i < R; i++) {
        printf("%d ", Available[i]);
    }
    printf("\n");

    printf("Allocation:\n");
    for (int i = 0; i < P; i++) {
        printf("P%d: ", i);
        for (int j = 0; j < R; j++) {
            printf("%d ", Allocation[i][j]);
        }
        printf("\n");
    }

    printf("Need:\n");
    for (int i = 0; i < P; i++) {
        printf("P%d: ", i);
        for (int j = 0; j < R; j++) {
            printf("%d ", Need[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// 安全性检查算法
bool isSafeState(const char* phase) {
    int Work[R];        // 工作向量
    bool Finish[P] = {false};  // 是否完成
    int safeSequence[P]; // 保存安全序列

    // 初始化 Work = Available
    for (int i = 0; i < R; i++) {
        Work[i] = Available[i];
    }

    int count = 0;

    printf("[%s] 开始安全性检查...\n", phase);
    while (count < P) {
        bool allocated = false;
        for (int i = 0; i < P; i++) {
            if (!Finish[i]) {
                bool canAllocate = true;

                for (int j = 0; j < R; j++) {
                    if (Need[i][j] > Work[j]) {
                        canAllocate = false;
                        break;
                    }
                }

                if (canAllocate) {
                    printf("  P%d 可以安全执行。释放资源后:\n", i);
                    for (int j = 0; j < R; j++) {
                        Work[j] += Allocation[i][j];
                    }
                    Finish[i] = true;
                    safeSequence[count++] = i;

                    printf("  Work: ");
                    for (int j = 0; j < R; j++) {
                        printf("%d ", Work[j]);
                    }
                    printf("\n");

                    allocated = true;
                }
            }
        }

        if (!allocated) {
            printf("[%s] 不安全状态，无法找到完整的安全序列。\n", phase);
            return false;
        }
    }

    // 打印安全序列
    printf("[%s] 系统安全。安全序列: ", phase);
    for (int i = 0; i < P; i++) {
        printf("P%d ", safeSequence[i]);
    }
    printf("\n");
    return true;
}

// 资源请求算法
bool requestResources(int processID, int request[], const char* phase) {
    printf("[%s] 进程 P%d 请求资源: ", phase, processID);
    for (int i = 0; i < R; i++) {
        printf("%d ", request[i]);
    }
    printf("\n");

    // 检查 Request <= Need 和 Request <= Available
    for (int i = 0; i < R; i++) {
        if (request[i] > Need[processID][i]) {
            printf("[%s] 请求超过需求，拒绝请求。\n", phase);
            return false;
        }
        if (request[i] > Available[i]) {
            printf("[%s] 请求超过可用资源，拒绝请求。\n", phase);
            return false;
        }
    }

    // 试探性分配资源
    printf("[%s] 试探性分配资源...\n", phase);
    for (int i = 0; i < R; i++) {
        Available[i] -= request[i];
        Allocation[processID][i] += request[i];
        Need[processID][i] -= request[i];
    }

    printSystemState(phase);

    // 检查分配后是否安全
    if (isSafeState(phase)) {
        printf("[%s] 请求被批准。\n", phase);
        return true;
    } else {
        // 回滚
        printf("[%s] 请求导致不安全状态，回滚资源分配。\n", phase);
        for (int i = 0; i < R; i++) {
            Available[i] += request[i];
            Allocation[processID][i] -= request[i];
            Need[processID][i] += request[i];
        }
        printSystemState("(回滚后)");
        return false;
    }
}