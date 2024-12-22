#include <stdio.h>
#include <stdbool.h>

// 页面请求数量应与数组中实际元素数相符
#define PAGE_REQUEST_COUNT 12 // 定义页面请求数量，需与实际页面数组匹配
int pageRequests[PAGE_REQUEST_COUNT] = {1, 4, 3, 1, 2, 5, 1, 4, 2, 1, 4, 5};

// 主存块数
#define FRAME_COUNT 3 // 定义主存块数量（帧数）

// 打印当前内存框中的页面状态
void printFrames(int frames[], int frameCount) {
    for (int i = 0; i < frameCount; i++) {
        if (frames[i] != -1) // 如果页面存在
            printf("%d ", frames[i]);
        else
            printf("- "); // 空帧以 "-" 表示
    }
    printf("\n");
}

// 检查页面是否在内存框中
bool isPageInFrames(int frames[], int frameCount, int page) {
    for (int i = 0; i < frameCount; i++) {
        if (frames[i] == page) // 遍历当前帧，查找页面
            return true; // 页面已在内存中，返回 true
    }
    return false; // 页面不在内存中，返回 false
}

// 找到需要替换的页面索引（FIFO算法核心）
int findFIFOIndex(int *pointer, int frameCount) {
    int index = *pointer; // 替换位置由指针决定
    *pointer = (*pointer + 1) % frameCount; // 更新替换指针，循环指向下一个帧
    return index; // 返回需要替换的页面索引
}

// 找到需要替换的页面索引（OPT算法核心）
int findOPTIndex(int frames[], int frameCount, int currentPageIndex) {
    int farthest = -1, replaceIndex = -1; // 初始化最远未来访问位置和替换索引
    for (int i = 0; i < frameCount; i++) { // 遍历当前帧
        int j;
        for (j = currentPageIndex + 1; j < PAGE_REQUEST_COUNT; j++) { 
            if (frames[i] == pageRequests[j]) { 
                if (j > farthest) { // 找到最远未来访问的页面
                    farthest = j;
                    replaceIndex = i;
                }
                break; // 页面已找到，停止内部循环
            }
        }
        if (j == PAGE_REQUEST_COUNT) { // 如果页面未来不再访问
            return i; // 直接返回该页面索引
        }
    }
    return replaceIndex; // 返回未来最晚访问的页面索引
}

// 找到需要替换的页面索引（LRU算法核心）
int findLRUIndex(int lastUsed[], int frameCount) {
    int min = lastUsed[0], index = 0; // 初始化最近最久未使用页面的时间和索引
    for (int i = 1; i < frameCount; i++) {
        if (lastUsed[i] < min) { // 查找时间最小的页面
            min = lastUsed[i];
            index = i;
        }
    }
    return index; // 返回最近最久未使用页面的索引
}

// 先进先出（FIFO）页面置换算法
void fifo() {
    printf("\n先进先出（FIFO）置换算法:\n");
    int frames[FRAME_COUNT]; // 定义内存框
    for (int i = 0; i < FRAME_COUNT; i++) frames[i] = -1; // 初始化内存框为空
    int pointer = 0, pageFaults = 0; // 替换指针和缺页计数器初始化

    for (int i = 0; i < PAGE_REQUEST_COUNT; i++) {
        if (!isPageInFrames(frames, FRAME_COUNT, pageRequests[i])) { 
            // 如果页面不在内存中，发生缺页
            int replaceIndex = findFIFOIndex(&pointer, FRAME_COUNT); // 找到替换位置
            frames[replaceIndex] = pageRequests[i]; // 替换页面
            pageFaults++; // 缺页次数加一
        }
        printf("访问页面 %d: ", pageRequests[i]);
        printFrames(frames, FRAME_COUNT); // 打印当前内存状态
    }
    printf("总缺页次数: %d\n", pageFaults);
}

// 最佳置换（OPT）页面置换算法
void opt() {
    printf("\n最佳置换（OPT）置换算法:\n");
    int frames[FRAME_COUNT]; // 定义内存框
    for (int i = 0; i < FRAME_COUNT; i++) frames[i] = -1; // 初始化内存框为空
    int pageFaults = 0; // 缺页计数器初始化

    for (int i = 0; i < PAGE_REQUEST_COUNT; i++) {
        if (!isPageInFrames(frames, FRAME_COUNT, pageRequests[i])) { 
            // 如果页面不在内存中，发生缺页
            int replaceIndex = findOPTIndex(frames, FRAME_COUNT, i); // 找到替换位置
            frames[replaceIndex] = pageRequests[i]; // 替换页面
            pageFaults++; // 缺页次数加一
        }
        printf("访问页面 %d: ", pageRequests[i]);
        printFrames(frames, FRAME_COUNT); // 打印当前内存状态
    }
    printf("总缺页次数: %d\n", pageFaults);
}

// 最近最久未使用（LRU）页面置换算法
void lru() {
    printf("\n最近最久未使用（LRU）置换算法:\n");
    int frames[FRAME_COUNT]; // 定义内存框
    for (int i = 0; i < FRAME_COUNT; i++) frames[i] = -1; // 初始化内存框为空
    int lastUsed[FRAME_COUNT] = {0, 0, 0}; // 初始化最近使用时间
    int pageFaults = 0, time = 0; // 缺页计数器和时间步初始化

    for (int i = 0; i < PAGE_REQUEST_COUNT; i++) {
        time++; // 时间步进
        if (!isPageInFrames(frames, FRAME_COUNT, pageRequests[i])) {
            // 页面不在内存中 -> 缺页
            int replaceIndex = findLRUIndex(lastUsed, FRAME_COUNT); // 找出LRU页面的索引
            frames[replaceIndex] = pageRequests[i]; // 用当前页面替换LRU页面
            lastUsed[replaceIndex] = time; // 更新新装入页面的最近使用时间
            pageFaults++; // 缺页次数加一
        } else {
            // 页面已在内存中 -> 命中
            // 需要更新该页面在 lastUsed 中的时间
            for (int j = 0; j < FRAME_COUNT; j++) {
                if (frames[j] == pageRequests[i]) {
                    lastUsed[j] = time; // 更新该页面的最近使用时间
                    break;
                }
            }
        }
        printf("访问页面 %d: ", pageRequests[i]);
        printFrames(frames, FRAME_COUNT); // 打印内存状态
    }
    printf("总缺页次数: %d\n", pageFaults);
}


int main() {
    opt();  // 运行 OPT 算法
    fifo(); // 运行 FIFO 算法
    lru();  // 运行 LRU 算法
    return 0;
}