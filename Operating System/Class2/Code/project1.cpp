#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// 定义 Job 结构体，用于表示作业的相关信息
struct Job {
    string name;     // 作业名
    int arrivalTime; // 到达时间
    int serviceTime; // 服务时间（执行时间）
    int startTime;   // 开始时间
    int finishTime;  // 完成时间
    int turnaroundTime; // 周转时间（完成时间 - 到达时间）
    double weightedTurnaroundTime; // 带权周转时间（周转时间 / 服务时间）
};

// 比较函数：按到达时间排序（用于先来先服务 FCFS 调度）
bool compareByArrivalTime(const Job& a, const Job& b) {
    return a.arrivalTime < b.arrivalTime;
}

// 比较函数：按服务时间排序（用于短作业优先 SJF 调度）
bool compareByServiceTime(const Job& a, const Job& b) {
    return a.serviceTime < b.serviceTime;
}

// 先来先服务调度算法（FCFS）
void FCFS(vector<Job>& jobs) {
    // 按到达时间排序
    sort(jobs.begin(), jobs.end(), compareByArrivalTime);

    int currentTime = 0; // 当前时间
    double totalTurnaroundTime = 0; // 总周转时间
    double totalWeightedTurnaroundTime = 0; // 总带权周转时间

    // 遍历每个作业，按照到达顺序处理
    for (auto& job : jobs) {
        // 作业开始运行的时间为当前时间或到达时间中的较大值
        job.startTime = max(currentTime, job.arrivalTime);
        job.finishTime = job.startTime + job.serviceTime; // 计算完成时间
        job.turnaroundTime = job.finishTime - job.arrivalTime; // 计算周转时间
        job.weightedTurnaroundTime = (double)job.turnaroundTime / job.serviceTime; // 计算带权周转时间

        // 累加总周转时间和总带权周转时间
        totalTurnaroundTime += job.turnaroundTime;
        totalWeightedTurnaroundTime += job.weightedTurnaroundTime;

        // 更新当前时间
        currentTime = job.finishTime;
    }

    // 输出结果
    cout << "FCFS 调度结果：" << endl;
    cout << "作业名 开始时间 完成时间 周转时间 带权周转时间" << endl;
    for (const auto& job : jobs) {
        cout << job.name << "      " << job.startTime << "        " 
             << job.finishTime << "        " << job.turnaroundTime << "       " 
             << job.weightedTurnaroundTime << endl;
    }

    // 输出平均周转时间和平均带权周转时间
    cout << "平均周转时间: " << totalTurnaroundTime / jobs.size() << endl;
    cout << "平均带权周转时间: " << totalWeightedTurnaroundTime / jobs.size() << endl;
}

// 短作业优先调度算法（SJF）
void SJF(vector<Job>& jobs) {
    // 按到达时间排序
    sort(jobs.begin(), jobs.end(), compareByArrivalTime);

    vector<Job> readyQueue; // 就绪队列，用于存储可运行的作业
    int currentTime = 0; // 当前时间
    double totalTurnaroundTime = 0; // 总周转时间
    double totalWeightedTurnaroundTime = 0; // 总带权周转时间
    int completedJobs = 0; // 完成的作业数量
    cout << "SJF 调度结果：" << endl;
    cout << "作业名 开始时间 完成时间 周转时间 带权周转时间" << endl;
    // 模拟调度过程
    while (!jobs.empty() || !readyQueue.empty()) {
        // 将已到达的作业加入就绪队列
        while (!jobs.empty() && jobs.front().arrivalTime <= currentTime) {
            readyQueue.push_back(jobs.front());
            jobs.erase(jobs.begin());
        }

        // 按服务时间排序，选择短作业优先
        sort(readyQueue.begin(), readyQueue.end(), compareByServiceTime);

        if (!readyQueue.empty()) {
            // 从就绪队列中选择短作业
            auto job = readyQueue.front();
            readyQueue.erase(readyQueue.begin());

            // 计算作业的开始时间、完成时间、周转时间和带权周转时间
            job.startTime = max(currentTime, job.arrivalTime);
            job.finishTime = job.startTime + job.serviceTime;
            job.turnaroundTime = job.finishTime - job.arrivalTime;
            job.weightedTurnaroundTime = (double)job.turnaroundTime / job.serviceTime;

            // 累加总周转时间和总带权周转时间
            totalTurnaroundTime += job.turnaroundTime;
            totalWeightedTurnaroundTime += job.weightedTurnaroundTime;
            completedJobs++; // 更新完成的作业数量

            // 更新当前时间
            currentTime = job.finishTime;

            // 输出作业的调度结果
            cout << job.name << "      " << job.startTime << "        "
                 << job.finishTime << "        " << job.turnaroundTime << "       "
                 << job.weightedTurnaroundTime << endl;
        } else {
            // 如果没有作业可运行，时间推进
            currentTime++;
        }
    }

    // 输出平均周转时间和平均带权周转时间
    if (completedJobs > 0) {
        cout << "平均周转时间: " << totalTurnaroundTime / completedJobs << endl;
        cout << "平均带权周转时间: " << totalWeightedTurnaroundTime / completedJobs << endl;
    }
}

int main() {
    // 初始化作业队列
    vector<Job> jobs = {
        {"A", 0, 4}, {"B", 1, 3}, {"C", 2, 5}, {"D", 3, 2}, {"E", 4, 4}
    };

    // 运行先来先服务调度算法（FCFS）
    vector<Job> jobsFCFS = jobs;
    FCFS(jobsFCFS);

    cout << endl;

    // 运行短作业优先调度算法（SJF）
    vector<Job> jobsSJF = jobs;
    SJF(jobsSJF);

    return 0;
}