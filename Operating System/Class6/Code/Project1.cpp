#include <stdio.h>
#define MAX_BLOCKS 100
#define MAX_PROCESSES 100

// 定义内存块结构，用于存储每个内存块的大小
typedef struct {
    int size;  // 内存块的大小
} MemoryBlock;

// 定义进程结构，用于存储每个进程的请求大小和已分配的内存块索引
typedef struct {
    int size;             // 进程请求的内存大小
    int allocated_block;  // 已分配的内存块索引，-1表示未分配
} Process;

// 实现首次适应算法（First Fit）进行内存分配
void first_fit(MemoryBlock blocks[], int num_blocks, Process processes[], int num_processes) {
    for (int i = 0; i < num_processes; i++) {
        processes[i].allocated_block = -1; // 初始化为未分配状态
        for (int j = 0; j < num_blocks; j++) {
            // 如果内存块的大小满足进程请求，则分配给当前进程
            if (blocks[j].size >= processes[i].size) {
                processes[i].allocated_block = j; // 记录分配的内存块索引
                blocks[j].size -= processes[i].size; // 更新内存块的剩余大小
                break; // 分配完成后退出循环
            }
        }
    }
}

// 打印内存分配结果及剩余内存块信息
void print_allocation(MemoryBlock blocks[], int num_blocks, Process processes[], int num_processes) {
    printf("\n进程\t请求大小\t分配内存块\n");
    for (int i = 0; i < num_processes; i++) {
        if (processes[i].allocated_block != -1) {
            // 如果已分配，输出分配的内存块编号（从1开始计数）
            printf("%d\t%d\t\t%d\n", i + 1, processes[i].size, processes[i].allocated_block + 1);
        } else {
            // 如果未分配，输出“无法分配”
            printf("%d\t%d\t\t无法分配\n", i + 1, processes[i].size);
        }
    }

    // 打印剩余内存块的大小
    printf("\n剩余内存块情况:\n");
    for (int i = 0; i < num_blocks; i++) {
        printf("内存块 %d: 剩余大小 = %d\n", i + 1, blocks[i].size);
    }
}

int main() {
    int num_blocks, num_processes;
    MemoryBlock blocks[MAX_BLOCKS]; // 定义空闲内存块数组
    Process processes[MAX_PROCESSES]; // 定义进程数组

    // 输入空闲内存块的数量
    printf("输入空闲内存块的数量: ");
    scanf("%d", &num_blocks);
    printf("输入每个内存块的大小:\n");
    for (int i = 0; i < num_blocks; i++) {
        printf("内存块 %d: ", i + 1);
        scanf("%d", &blocks[i].size); // 输入每个内存块的大小
    }

    // 输入进程请求的内存大小
    printf("输入进程的数量: ");
    scanf("%d", &num_processes);
    printf("输入每个进程的请求大小:\n");
    for (int i = 0; i < num_processes; i++) {
        printf("进程 %d: ", i + 1);
        scanf("%d", &processes[i].size); // 输入每个进程请求的内存大小
    }

    // 执行首次适应算法进行内存分配
    first_fit(blocks, num_blocks, processes, num_processes);

    // 输出分配结果和剩余内存块情况
    print_allocation(blocks, num_blocks, processes, num_processes);

    return 0;
}