#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>
using namespace std;
#define NUM_WINDOWS 4  //银行默认工作窗口数量
#define SIMULATION_DURATION 510 //银行工作时间
#define MAX_ARRIVAL_INTERVAL 10 // 最大到达间隔
#define MIN_SERVICE_TIME 5 // 最小服务时间
#define MAX_SERVICE_TIME 15 // 最大服务时间
#define START_HOUR 8 // 开门的起始小时
#define START_MINUTE 30 // 开门的起始分钟
#define LUNCH_START_HOUR 12 // 午休开始的小时
#define LUNCH_END_HOUR 14 // 午休结束的小时
#define END_HOUR 17 // 关门的小时
#define NUM_WINDOWS_LUNCH 1 // 午休时间工作的窗口数量

typedef struct Customer {
    int id; // 顾客序号
    int arrival_time; // 到达时刻
    int service_time; // 需要服务的时长
} Customer, ElemType;

typedef struct Node {
    Customer data;
    Node* next;
} Node;

typedef struct Queue {
    Node* head;
    Node* tail;
    int size;
} Queue;

void initQueue(Queue* q) {
    q->head = NULL;
    q->tail = NULL;
    q->size = 0;
}

void enqueue(Queue* q, Customer customer) {
    Node* newNode = new Node;
    newNode->data = customer;
    newNode->next = NULL;
    if (q->head == NULL) {
        q->head = newNode;
        q->tail = newNode;
    }
    else {
        q->tail->next = newNode;
        q->tail = newNode;
    }
    q->size++;
}

void dequeue(Queue* q) {
    if (q->head != NULL) {
        Node* temp = q->head;
        q->head = q->head->next;
        delete temp;
        if (q->head == NULL) {
            q->tail = NULL;
        }
        q->size--;
    }
}

Customer getQueueHead(Queue* q) {
    if (q->head != NULL) {
        return q->head->data;
    }
    else {
        Customer emptyCustomer = { 0, 0, 0 };
        return emptyCustomer; // 返回一个空的顾客结构体
    }
}

typedef struct Window {
    int cur_remaining_service_time; // 当前正在办理业务的客户的剩余服务时长
    int remaining_service_time; // 总剩余服务时长
    int total_wait_time; // 统计该服务窗口所有顾客的等待时长
    Queue queue; // 服务窗口队列
} Window;

typedef struct Bank {
    int open_time, close_time; // 银行上下班时间
    Window windows[NUM_WINDOWS]; // 几个服务窗口
    Queue customer_queue; // 已生成的顾客队列
    int num_to_service; // 需要服务的总人数
} Bank;

void OpenForDay(Bank* bank);
void genCustomers(Bank* bank); // 生成所有顾客
void Clock(Bank* bank); // 时间推移函数
void EventDriven(Bank* bank, Customer* cur_customer, int now);
double CloseForDay(Bank* bank);
void printStatus(Bank* bank, int now);
bool allWindowsIdle(Bank* bank);
void setSeed(int seed);
int genRand(int min, int max);


void setSeed(int seed = -1) { // -1使用时间作为随机种子
    if (seed == -1) {
        srand(time(NULL));
    }
    else {
        srand(seed);
    }
}

int genRand(int min, int max) {
    return min + rand() % (max - min + 1);
}

void OpenForDay(Bank* bank)
{ // 初始化函数
    bank->open_time = 0;
    bank->close_time = SIMULATION_DURATION;
    for (int i = 0; i < NUM_WINDOWS; ++i) { // 初始化服务窗口
        bank->windows[i].cur_remaining_service_time = 0;
        bank->windows[i].remaining_service_time = 0;
        bank->windows[i].total_wait_time = 0;
        initQueue(&bank->windows[i].queue); // 初始化服务窗口队列
    }
    initQueue(&bank->customer_queue); // 初始化已生成的顾客队列
    bank->num_to_service = 0; // 已服务的顾客数设为0
    genCustomers(bank);
}

void genCustomers(Bank* bank)
{
    // 生成的顾客应该分布在整个模拟时间内
    int customer_id = 1; // 开始的顾客ID
    for (int t = bank->open_time; t < bank->close_time; t += genRand(1, MAX_ARRIVAL_INTERVAL)) {
        Customer customer = {
            customer_id++,
            t,
            genRand(MIN_SERVICE_TIME, MAX_SERVICE_TIME)
        };
        enqueue(&bank->customer_queue, customer);
        cout << "Customer " << customer.id << " Arrived at " << customer.arrival_time << " Estimated Service Time is " << customer.service_time << endl;
    }
}

void Clock(Bank* bank) {
    int now = 0; // 初始化当前时间
    // 继续运行，直到所有顾客都被服务完毕
    while (now < bank->close_time || !allWindowsIdle(bank) || bank->customer_queue.size > 0) {
        // 服务窗口更新，只在银行开放时间内接待新顾客
        for (int i = 0; i < NUM_WINDOWS; ++i) {
            if (bank->windows[i].remaining_service_time > 0) {
                bank->windows[i].remaining_service_time -= 1;
            }
            if (bank->windows[i].cur_remaining_service_time > 0) {
                bank->windows[i].cur_remaining_service_time -= 1;
                if (bank->windows[i].cur_remaining_service_time == 0 && bank->windows[i].queue.size > 0) {
                    Customer next_customer = getQueueHead(&bank->windows[i].queue);
                    bank->windows[i].cur_remaining_service_time = next_customer.service_time;
                    dequeue(&bank->windows[i].queue); // 从队列中移除已经开始服务的客户
                }
            }
        }

        // 在银行开放时间内接待新顾客
        if (now < bank->close_time) {
            while (bank->customer_queue.size > 0 && getQueueHead(&bank->customer_queue).arrival_time <= now) {
                Customer cur_customer = getQueueHead(&bank->customer_queue);
                dequeue(&bank->customer_queue); // 从顾客队列中移除该顾客
                EventDriven(bank, &cur_customer, now);
            }
        }
        // 检查是否所有窗口都空闲，且没有顾客在排队
        if (allWindowsIdle(bank) && bank->customer_queue.size == 0 && now >= bank->close_time) {
            break; // 如果是，跳出循环
        }

        // 打印当前状态
        printStatus(bank, now);
        now++; // 时间前进
    }
}

bool allWindowsIdle(Bank* bank) {
    for (int i = 0; i < NUM_WINDOWS; ++i) {
        if (bank->windows[i].remaining_service_time > 0 || bank->windows[i].queue.size > 0) {
            return false; // 如果有窗口不空闲或有顾客在排队，返回 false
        }
    }
    return true; // 所有窗口空闲，没有顾客在排队
}



void EventDriven(Bank* bank, Customer* cur_customer, int now)
{
    int min_index = 0; // 用于跟踪最短队列的窗口索引
    int min_size = INT_MAX; // 设置一个较大的数作为比较基准
    for (int i = 0; i < NUM_WINDOWS; ++i) {
        if (bank->windows[i].queue.size < min_size) {
            min_index = i;
            min_size = bank->windows[i].queue.size;
        }
    }
    int min = min_index;
    // 顾客入队
    enqueue(&bank->windows[min].queue, *cur_customer);
    bank->num_to_service += 1; // 总顾客数+1
    // 统计时长
    if (bank->windows[min].remaining_service_time > 0) { // 1. 统计等待时长；要考虑到每个人的等待时长，所以要累加剩余服务时长
        bank->windows[min].total_wait_time += bank->windows[min].remaining_service_time;
    }
    bank->windows[min].total_wait_time += cur_customer->service_time; // 2. 统计服务时长
    if (bank->windows[min].remaining_service_time == 0) {
        bank->windows[min].cur_remaining_service_time = cur_customer->service_time; // 3. 总剩余服务时长为0，说明该客户为当前队列“第一个”客户
    }
    bank->windows[min].remaining_service_time += cur_customer->service_time; // 4. 刷新该队列的剩余服务时长
}

double CloseForDay(Bank* bank)
{ // 计算客户平均逗留时长
    double total_time = 0;
    for (int i = 0; i < NUM_WINDOWS; ++i) {
        total_time += bank->windows[i].total_wait_time;
    }
    for (int i = 0; i < NUM_WINDOWS; ++i) {
        // 如果队列中还有客户，则继续累加他们的服务时间
        Node* p = bank->windows[i].queue.head;
        while (p != NULL) {
            total_time += bank->close_time - p->data.arrival_time;
            p = p->next;
        }
    }
    while (bank->customer_queue.size > 0) {
        dequeue(&bank->customer_queue);
    }
    return total_time / bank->num_to_service; // 假设关门时没有办理完成的顾客继续办理直到完成
}


void printStatus(Bank* bank, int now) {
    // 将模拟的分钟数转换为绝对时间
    int totalMinutes = now + START_HOUR * 60 + START_MINUTE; // 加上模拟开始的实际分钟数
    int hours = totalMinutes / 60; // 计算小时数
    int minutes = totalMinutes % 60; // 计算分钟数

    hours = hours % 24;

    // 输出当前时间，格式为24小时制
    cout << "=====CLOCK[" << setw(2) << setfill('0') << hours << ":" << setw(2) << setfill('0') << minutes << "]=====" << endl;
    for (int i = 0; i < NUM_WINDOWS; ++i) {
        Node* p = bank->windows[i].queue.head;
        cout << "Queue[" << i << "] ";
        while (p != NULL) {
            cout << p->data.id << " ";
            p = p->next;
        }
        cout << "[TAIL]" << endl;
    }
}

int main()
{
    //setSeed(2023);
    Bank bank;
    OpenForDay(&bank); // 开门
    Clock(&bank); // 时间流逝 -> 事件驱动
    cout << "Average Staying Time per Customer: " << CloseForDay(&bank) << endl; // 关门
    getchar();
    return 0;
}