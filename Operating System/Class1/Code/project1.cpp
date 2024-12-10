#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// PCB 结构定义，表示进程控制块
typedef struct PCB {
    int pid;                  // 进程 ID
    char state[10];           // 进程状态 (READY, RUNNING, 等等)
    int priority;             // 进程优先级
    int cpu_time;             // 进程需要的 CPU 时间（服务时间）
    int remaining_time;       // 剩余的服务时间
    int arrival_time;         // 进程到达时间
    int completion_time;      // 进程完成时间
    int turnaround_time;      // 进程周转时间
    float weighted_turnaround_time; // 进程带权周转时间
    struct PCB *next;         // 指向下一个 PCB 的指针
} PCB;

// 就绪队列结构定义，存储所有等待被调度的进程
typedef struct ReadyQueue {
    PCB *head;                // 指向就绪队列的头节点
    PCB *tail;                // 指向就绪队列的尾节点
} ReadyQueue;

// 初始化就绪队列，返回一个空的队列指针
ReadyQueue* initQueue() {
    ReadyQueue *queue = (ReadyQueue *)malloc(sizeof(ReadyQueue));
    queue->head = NULL;       // 队列头为空
    queue->tail = NULL;       // 队列尾为空
    return queue;
}

// 向就绪队列中添加进程 (尾部插入)
void enqueue(ReadyQueue *queue, PCB *process) {
    if (queue->tail == NULL) {        // 如果队列为空
        queue->head = process;       // 新节点既是头，也是尾
        queue->tail = process;
    } else {
        queue->tail->next = process; // 插入到尾部
        queue->tail = process;       // 更新队列尾
    }
    process->next = NULL;            // 确保新节点的 next 为 NULL
}

// 从就绪队列中移除队首进程
PCB* dequeue(ReadyQueue *queue) {
    if (queue->head == NULL) return NULL; // 如果队列为空，返回 NULL

    PCB *process = queue->head;          // 取队首节点
    queue->head = process->next;         // 更新队列头
    if (queue->head == NULL) queue->tail = NULL; // 如果队列为空，更新尾指针

    return process;
}

// 创建一个新的 PCB 并初始化
PCB* createPCB(int pid, int priority, int cpu_time, int arrival_time) {
    PCB *process = (PCB *)malloc(sizeof(PCB));
    process->pid = pid;                        // 设置进程 ID
    snprintf(process->state, sizeof(process->state), "READY"); // 初始化状态为 READY
    process->priority = priority;              // 设置优先级
    process->cpu_time = cpu_time;              // 设置 CPU 时间
    process->remaining_time = cpu_time;        // 剩余时间初始化为服务时间
    process->arrival_time = arrival_time;      // 设置到达时间
    process->completion_time = 0;              // 初始化完成时间
    process->turnaround_time = 0;              // 初始化周转时间
    process->weighted_turnaround_time = 0.0;   // 初始化带权周转时间
    process->next = NULL;                      // 初始化为没有后续节点
    return process;
}

// 优先级调度算法实现
void priorityScheduling(ReadyQueue *queue) {
    int current_time = 0;                      // 当前时间初始化为 0
    int total_turnaround_time = 0;             // 总周转时间
    float total_weighted_turnaround_time = 0;  // 总带权周转时间
    int process_count = 0;                     // 记录进程总数

    printf("\nPriority Scheduling:\n");
    printf("PID\tArrival\tService\tPriority\tCompletion\tTurnaround\tWeighted Turnaround\n");

    while (queue->head != NULL) {              // 如果队列不为空
        PCB *highestPriorityProcess = NULL;    // 当前时刻可调度的最高优先级进程
        PCB *prev = NULL, *current = queue->head, *highestPrev = NULL;

        // 找到当前时刻已到达且优先级最高的进程
        while (current != NULL) {
            if (current->arrival_time <= current_time) { // 只考虑已到达的进程
                if (highestPriorityProcess == NULL || current->priority > highestPriorityProcess->priority) {
                    highestPriorityProcess = current;
                    highestPrev = prev;
                }
            }
            prev = current;
            current = current->next;
        }

        // 如果当前时刻没有进程可以调度，推进时间
        if (highestPriorityProcess == NULL) {
            current_time++;
            continue;
        }

        // 从队列中移除最高优先级进程
        if (highestPrev == NULL) {
            queue->head = highestPriorityProcess->next;
        } else {
            highestPrev->next = highestPriorityProcess->next;
        }
        if (highestPriorityProcess == queue->tail) {
            queue->tail = highestPrev;
        }

        // 更新完成时间和其他信息
        current_time += highestPriorityProcess->cpu_time;
        highestPriorityProcess->completion_time = current_time;
        highestPriorityProcess->turnaround_time = current_time - highestPriorityProcess->arrival_time;
        highestPriorityProcess->weighted_turnaround_time = (float)highestPriorityProcess->turnaround_time / highestPriorityProcess->cpu_time;

        // 输出进程信息
        printf("%d\t%d\t%d\t%d\t\t%d\t\t%d\t\t%.2f\n",
               highestPriorityProcess->pid,
               highestPriorityProcess->arrival_time,
               highestPriorityProcess->cpu_time,
               highestPriorityProcess->priority,
               highestPriorityProcess->completion_time,
               highestPriorityProcess->turnaround_time,
               highestPriorityProcess->weighted_turnaround_time);

        total_turnaround_time += highestPriorityProcess->turnaround_time;
        total_weighted_turnaround_time += highestPriorityProcess->weighted_turnaround_time;
        process_count++;

        free(highestPriorityProcess);
    }

    // 输出平均值
    printf("\nAverage Turnaround Time: %.2f\n", (float)total_turnaround_time / process_count);
    printf("Average Weighted Turnaround Time: %.2f\n", total_weighted_turnaround_time / process_count);
}

// 时间片轮转调度算法实现
// 实现基于时间片的进程调度，确保所有进程能公平地获取 CPU 时间
void roundRobinScheduling(ReadyQueue *queue, int time_slice) {
    int current_time = 0; // 当前时间初始化为 0
    int total_turnaround_time = 0; // 用于累积所有进程的周转时间
    float total_weighted_turnaround_time = 0; // 用于累积所有进程的带权周转时间
    int process_count = 0; // 记录调度的进程数量

    // 打印表头，显示每列数据的含义
    printf("\nTime Slice Round Robin Scheduling:\n");
    printf("PID\tArrival\tService\tCompletion\tTurnaround\tWeighted Turnaround\n");

    // 循环调度，直到就绪队列为空
    while (queue->head != NULL) {
        PCB *process = dequeue(queue); // 从队列中移除队首进程

        // 如果当前时间小于进程的到达时间，跳跃到进程到达时间
        if (current_time < process->arrival_time) {
            current_time = process->arrival_time;
        }

        // 如果剩余服务时间大于时间片，则继续执行一轮调度
        if (process->remaining_time > time_slice) {
            current_time += time_slice; // 当前时间增加一个时间片
            process->remaining_time -= time_slice; // 减少剩余服务时间
            enqueue(queue, process); // 将进程重新加入队尾
        } else {
            // 如果剩余服务时间小于或等于时间片，进程完成执行
            current_time += process->remaining_time; // 当前时间增加进程剩余时间
            process->remaining_time = 0; // 剩余时间设置为 0
            process->completion_time = current_time; // 记录进程完成时间
            process->turnaround_time = process->completion_time - process->arrival_time; // 计算周转时间
            process->weighted_turnaround_time = (float)process->turnaround_time / process->cpu_time; // 计算带权周转时间

            // 输出进程调度信息
            printf("%d\t%d\t%d\t%d\t\t%d\t\t%.2f\n",
                   process->pid,
                   process->arrival_time,
                   process->cpu_time,
                   process->completion_time,
                   process->turnaround_time,
                   process->weighted_turnaround_time);

            // 累积总的周转时间和带权周转时间
            total_turnaround_time += process->turnaround_time;
            total_weighted_turnaround_time += process->weighted_turnaround_time;
            process_count++; // 增加已完成进程的计数

            free(process); // 释放完成的进程所占用的内存
        }
    }

    // 计算并输出平均周转时间和平均带权周转时间
    printf("\nAverage Turnaround Time: %.2f\n", (float)total_turnaround_time / process_count);
    printf("Average Weighted Turnaround Time: %.2f\n", total_weighted_turnaround_time / process_count);
}

// 主函数
int main() {
    ReadyQueue *queue1 = initQueue();

    enqueue(queue1, createPCB(1, 10, 50, 0));
    enqueue(queue1, createPCB(2, 20, 30, 2));
    enqueue(queue1, createPCB(3, 15, 20, 4));

    priorityScheduling(queue1);

    ReadyQueue *queue2 = initQueue();

    enqueue(queue2, createPCB(1, 10, 50, 0));
    enqueue(queue2, createPCB(2, 20, 30, 2));
    enqueue(queue2, createPCB(3, 15, 20, 4));
    int time_slice = 10;

    roundRobinScheduling(queue2, time_slice);

    return 0;
}