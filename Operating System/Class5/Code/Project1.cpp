#include <stdio.h>
#include <stdlib.h>

#define MAX_REQUESTS 100

int find_closest_request(int current_position, int requests[], int visited[], int n) {
    int min_distance = 10000;
    int closest_index = -1;
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            int distance = abs(requests[i] - current_position);
            if (distance < min_distance) {
                min_distance = distance;
                closest_index = i;
            }
        }
    }
    return closest_index;
}

void sstf(int start, int requests[], int n) {
    int visited[MAX_REQUESTS] = {0};
    int current_position = start;
    int total_seek_distance = 0;

    printf("SSTF 调度顺序: ");
    for (int i = 0; i < n; i++) {
        int closest_index = find_closest_request(current_position, requests, visited, n);
        visited[closest_index] = 1;

        int seek_distance = abs(requests[closest_index] - current_position);
        total_seek_distance += seek_distance;

        printf("%d ", requests[closest_index]);
        printf("(移动距离: %d)\n", seek_distance);

        current_position = requests[closest_index];
    }

    printf("\n总寻道距离: %d\n", total_seek_distance);
    printf("平均寻道距离: %.2f\n", (float)total_seek_distance / n);
}

void scan(int start, int requests[], int n, int direction) {
    int sorted_requests[MAX_REQUESTS];
    for (int i = 0; i < n; i++) {
        sorted_requests[i] = requests[i];
    }

    // 排序请求
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (sorted_requests[j] > sorted_requests[j + 1]) {
                int temp = sorted_requests[j];
                sorted_requests[j] = sorted_requests[j + 1];
                sorted_requests[j + 1] = temp;
            }
        }
    }

    int total_seek_distance = 0;
    int current_position = start;

    printf("SCAN 调度顺序: \n");
    if (direction == 1) {  // 向上方向
        // 仅处理大于当前磁道的请求
        for (int i = 0; i < n; i++) {
            if (sorted_requests[i] >= current_position) {
                int seek_distance = abs(sorted_requests[i] - current_position);
                total_seek_distance += seek_distance;

                printf("%d (移动距离: %d)\n", sorted_requests[i], seek_distance);
                current_position = sorted_requests[i];
            }
        }
        // 处理反方向的请求
        for (int i = n - 1; i >= 0; i--) {
            if (sorted_requests[i] < start) {
                int seek_distance = abs(sorted_requests[i] - current_position);
                total_seek_distance += seek_distance;

                printf("%d (移动距离: %d)\n", sorted_requests[i], seek_distance);
                current_position = sorted_requests[i];
            }
        }
    } else {  // 向下方向
        // 仅处理小于当前磁道的请求
        for (int i = n - 1; i >= 0; i--) {
            if (sorted_requests[i] <= current_position) {
                int seek_distance = abs(sorted_requests[i] - current_position);
                total_seek_distance += seek_distance;

                printf("%d (移动距离: %d)\n", sorted_requests[i], seek_distance);
                current_position = sorted_requests[i];
            }
        }
        // 处理反方向的请求
        for (int i = 0; i < n; i++) {
            if (sorted_requests[i] > start) {
                int seek_distance = abs(sorted_requests[i] - current_position);
                total_seek_distance += seek_distance;

                printf("%d (移动距离: %d)\n", sorted_requests[i], seek_distance);
                current_position = sorted_requests[i];
            }
        }
    }

    printf("\n总寻道距离: %d\n", total_seek_distance);
    printf("平均寻道距离: %.2f\n", (float)total_seek_distance / n);
}

int main() {
    int requests[] = {150, 160, 184, 18, 38, 39, 55, 58, 90}; // 请求序列
    int n = sizeof(requests) / sizeof(requests[0]);           // 请求数量
    int start_position = 100;                                // 起始位置

    printf("========== SSTF ==========\n");
    sstf(start_position, requests, n);

    printf("\n========== SCAN 向上==========\n");
    scan(start_position, requests, n, 1); // 1 表示向上方向

    printf("\n========== SCAN 向下==========\n");
    scan(start_position, requests, n, 0); // 0 表示向上方向

    return 0;
}