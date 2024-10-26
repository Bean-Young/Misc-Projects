#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace std;
#define NUM_WINDOWS 3 //银行默认工作窗口数量
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
const vector<string> daysOfWeek = { "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday" };
int leave;
double staytime;
typedef struct Customer {
    int id;
    int arrival_time;
    int service_time;
    int tolerance_time;
    int waiting_time = 0; // 新增等待时间
    bool is_served = true; // 新增是否被服务标记
} Customer;


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
    int cur_customer_id = 0; // 当前正在服务的顾客ID，0表示没有顾客
    int cur_remaining_service_time = 0; // 当前正在办理业务的顾客的剩余服务时长
} Window;


typedef struct Bank {
    int open_time, close_time;
    Window windows[NUM_WINDOWS];
    Queue waitingQueue;
    Queue customer_queue;
    int num_to_service;
    vector<Customer> daily_customers; // 新增顾客数组
    vector<int> leave_customers;
} Bank;

void OpenForDay(Bank* bank);
void genCustomers(Bank* bank); // 生成所有顾客
void Clock(Bank* bank); // 时间推移函数
void EventDriven(Bank* bank, Customer* cur_customer, int now);
void CloseForDay(Bank* bank);
void printStatus(Bank* bank, int now, int active_windows);
bool allWindowsIdle(Bank* bank);
void setSeed(int seed);
int genRand(int min, int max);
void transferFromWaitingQueueToWindow(Bank* bank,int active_windows);

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

void OpenForDay(Bank* bank) {
    // 初始化函数
    bank->open_time = START_HOUR * 60 + START_MINUTE; // 使用绝对时间初始化银行的开门时间
    bank->close_time = bank->open_time + SIMULATION_DURATION; // 假定SIMULATION_DURATION是从开门到关门的持续时间
    for (int i = 0; i < NUM_WINDOWS; ++i) {
        // 初始化服务窗口
        bank->windows[i].cur_customer_id = 0; // 窗口开始时没有正在服务的顾客
        bank->windows[i].cur_remaining_service_time = 0; // 没有顾客在服务，所以剩余服务时间为0
    }
    initQueue(&bank->waitingQueue); // 初始化等候区队列
    initQueue(&bank->customer_queue); // 初始化已生成的顾客队列
    bank->num_to_service = 0; // 重置已服务的顾客数量为0
    genCustomers(bank); // 生成顾客
}

void genCustomers(Bank* bank) {
    int start_minutes = START_HOUR * 60 + START_MINUTE;
    int customer_id = 1;
    for (int t = 0; t < SIMULATION_DURATION; t += genRand(1, MAX_ARRIVAL_INTERVAL)) {
        int absolute_arrival_time = start_minutes + t;

        int tolerance_time = genRand(MIN_SERVICE_TIME, MAX_SERVICE_TIME * 2);

        Customer customer = {
            customer_id++,
            absolute_arrival_time,
            genRand(MIN_SERVICE_TIME, MAX_SERVICE_TIME),
            tolerance_time,
            0, // 初始化等待时间为0
            true // 初始化被服务
        };

        bank->daily_customers.push_back(customer); // 将顾客添加到数组中
        enqueue(&bank->customer_queue, customer); // 也将顾客加入到队列中
        bank->num_to_service++;

        // 打印顾客到达信息，包括到达时间和预计服务时间
        int hours = absolute_arrival_time / 60;
        int minutes = absolute_arrival_time % 60;
        cout << "Customer " << customer.id << " Arrived at "
            << setw(2) << setfill('0') << hours << ":"
            << setw(2) << setfill('0') << minutes
            << " Estimated Service Time is " << customer.service_time
            << " Tolerance Time is " << customer.tolerance_time << endl;
    }
}

void transferFromWaitingQueueToWindow(Bank* bank, int active_windows) {
    for (int i = 0; i < active_windows; ++i) {
        if (bank->windows[i].cur_customer_id == 0 && bank->waitingQueue.size > 0) {
            Customer customer = getQueueHead(&bank->waitingQueue);
            bank->windows[i].cur_customer_id = customer.id;
            bank->windows[i].cur_remaining_service_time = customer.service_time;
            dequeue(&bank->waitingQueue);
        }
    }
}


void Clock(Bank* bank) {
    int now = START_HOUR * 60 + START_MINUTE;
    int lunch_start_time = LUNCH_START_HOUR * 60;
    int lunch_end_time = LUNCH_END_HOUR * 60;
    bool is_lunch_time = false;

    while (now < bank->close_time || !allWindowsIdle(bank) || bank->waitingQueue.size > 0) {
        is_lunch_time = now >= lunch_start_time && now < lunch_end_time;
        // 处理等待队列中顾客的等待时间和容忍时间
        Node* current = bank->waitingQueue.head;
        Node* prev = nullptr;
        while (current != nullptr) {
            if (current->data.is_served) {
                current->data.waiting_time++;
                // 更新daily_customers中对应顾客的等待时间
                for (auto& customer : bank->daily_customers) {
                    if (customer.id == current->data.id) {
                        customer.waiting_time = current->data.waiting_time;
                        break;
                    }
                }
                if (current->data.waiting_time > current->data.tolerance_time) {
                    // 如果等待时间超过容忍时间，顾客离开
                    bank->leave_customers.push_back(current->data.id);
                    current->data.is_served = false;
                    for (auto& customer : bank->daily_customers) {
                        if (customer.id == current->data.id) {
                            customer.is_served = current->data.is_served;
                            break;
                        }
                    }
                    Node* toDelete = current;
                    if (prev != nullptr) {
                        prev->next = current->next;
                    }
                    else {
                        bank->waitingQueue.head = current->next;
                    }
                    if (current == bank->waitingQueue.tail) {
                        bank->waitingQueue.tail = prev;
                    }
                    current = current->next;
                    delete toDelete;
                    bank->waitingQueue.size--;
                    continue;
                }
            }
            prev = current;
            current = current->next;
        }
        int active_windows = is_lunch_time ? NUM_WINDOWS_LUNCH : NUM_WINDOWS;
        int working_during_lunch = 0;
        if (is_lunch_time) {
            // 如果是午休时间，需要检查哪些窗口可以继续工作
            for (int i = active_windows; i < NUM_WINDOWS; i++) {
                if (bank->windows[i].cur_remaining_service_time > 0) {
                    working_during_lunch=i;
                }
            }
        }
        // 处理新到达的顾客
        while (bank->customer_queue.size > 0 && getQueueHead(&bank->customer_queue).arrival_time <= now) {
            Customer customer = getQueueHead(&bank->customer_queue);
            bool assigned = false;
            for (int i = 0; i < active_windows; ++i) {
                if (bank->windows[i].cur_customer_id == 0) { // 检查是否有空窗口
                    bank->windows[i].cur_customer_id = customer.id;
                    bank->windows[i].cur_remaining_service_time = customer.service_time;
                    dequeue(&bank->customer_queue); // 从顾客队列中移除该顾客
                    assigned = true;
                    break;
                }
            }
            if (!assigned) {
                enqueue(&bank->waitingQueue, customer); // 如果所有窗口都忙，将顾客加入等待队列
                dequeue(&bank->customer_queue); // 从顾客队列中移除
            }
        }

        // 更新所有窗口中顾客的服务时间
        for (int i = 0; i < NUM_WINDOWS; ++i) {
            if (bank->windows[i].cur_remaining_service_time > 0) {
                bank->windows[i].cur_remaining_service_time--;
                if (bank->windows[i].cur_remaining_service_time == 0) {
                    bank->windows[i].cur_customer_id = 0; // 完成服务
                }
            }
        }

        // 检查是否有顾客可以从等待队列移动到服务窗口
        transferFromWaitingQueueToWindow(bank, active_windows);
        printStatus(bank, now, active_windows+working_during_lunch);
        now++; // 时间前进一分钟
    }
}


bool allWindowsIdle(Bank* bank) {
    for (int i = 0; i < NUM_WINDOWS; ++i) {
        if (bank->windows[i].cur_remaining_service_time > 0) {
            return false; // 如果有窗口不空闲，返回 false
        }
    }
    return true; // 所有窗口空闲
}


void CloseForDay(Bank* bank) {
    double total_time_in_bank = 0.0;
    int customers_served_or_waited = 0;

    for (const Customer& customer : bank->daily_customers) {
        if (customer.is_served || customer.waiting_time > 0) {
            // 累加等待时间和（对于被服务的顾客）服务时间
            total_time_in_bank += customer.waiting_time + (customer.is_served ? customer.service_time : 0);
            customers_served_or_waited++;
        }
    }

    double average_time = customers_served_or_waited > 0 ? total_time_in_bank / customers_served_or_waited : 0.0;
    staytime += average_time;
    cout << "Average Staying Time per Customer: " << average_time << " minutes" << endl;

    // 打印离开顾客的数量和ID
    leave += bank->leave_customers.size();
    cout << "Number of Leaving Customers: " << bank->leave_customers.size() << endl;
    cout << "ID of Leaving Customers: ";
    for (int id : bank->leave_customers) {
        cout << id << " ";
    }
    cout << endl;

    // 清空等待队列和每日顾客数组
    while (bank->waitingQueue.size > 0) {
        dequeue(&bank->waitingQueue);
    }
    bank->daily_customers.clear(); // 清空顾客数组
    bank->leave_customers.clear(); // 清空离开顾客数组
}


void printStatus(Bank* bank, int now, int active_windows) {
    int hours = now / 60;
    int minutes = now % 60;
    cout << "=====CLOCK[" << setw(2) << setfill('0') << hours << ":" << setw(2) << setfill('0') << minutes << "] ";
    cout << (now >= LUNCH_START_HOUR * 60 && now < LUNCH_END_HOUR * 60 ? "Lunch Time" : "Normal Hours") << " =====" << endl;

    for (int i = 0; i < active_windows; ++i) {
        cout << "Window[" << i << "] ";
        if (bank->windows[i].cur_customer_id != 0) {
            cout << "is serving Customer ID: " << bank->windows[i].cur_customer_id;
        }
        else {
            cout << "is empty";
        }
        cout << endl;
    }

    // 打印等候区状态
    if (bank->waitingQueue.size > 0) {
        cout << "Waiting Queue: ";
        Node* p = bank->waitingQueue.head;
        while (p != NULL) {
            if (std::find(bank->leave_customers.begin(), bank->leave_customers.end(), p->data.id) == bank->leave_customers.end()) {
                cout << p->data.id << " ";
            }
            p = p->next;
        }
        cout << endl;
    }
}

int main()
{
    Bank bank;
    for (int i = 1;i < 7;i++)
    {
        cout << "------------------------------------------Today is " << daysOfWeek[i - 1] <<"------------------------------------------" <<endl;
        setSeed(2023+i);
        OpenForDay(&bank); // 开门
        Clock(&bank); // 时间流逝 -> 事件驱动
        CloseForDay(&bank); // 关门
    }
    cout << "------------------------------------------Today is " << daysOfWeek[6] << "------------------------------------------" << endl;
    cout << "Average Staying Time per Customer(This Week): " << staytime/6 << " minutes" << endl;
    cout << "Number of Leaving Customers(This Week): " << leave << endl;
    getchar();
    return 0;
}