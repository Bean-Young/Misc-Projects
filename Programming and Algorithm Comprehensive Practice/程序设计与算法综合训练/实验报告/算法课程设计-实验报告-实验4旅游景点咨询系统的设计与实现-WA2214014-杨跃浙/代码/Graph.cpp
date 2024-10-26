#include <iostream>
#include <string>
#include <limits>
using namespace std;

#define _CRT_SECURE_NO_WARNINGS 1
#define MAXSIZE 100
#define INF numeric_limits<int>::max()

typedef struct {
    string* Vertex;
    int Matrix[MAXSIZE][MAXSIZE];
    int EdgeNum, VexNum;
} MGraph;

void InitMGraph(MGraph* g) {
    g->VexNum = 13;
    g->EdgeNum = 18;
    string* s = new string[13]{
        "玉屏索道入口",
        "玉屏索道出口",
        "天都峰",
        "迎客松",
        "排云溪站",
        "探海亭",
        "排云亭",
        "光明顶",
        "松谷庵站",
        "白鹤岭",
        "始信峰",
        "云谷索道入口",
        "云谷索道出口"
    };

    g->Vertex = s;

    for (int i = 0; i < g->VexNum; i++) {
        for (int j = 0; j < g->VexNum; j++) {
            g->Matrix[i][j] = INF;
        }
    }
    g->Matrix[0][1] = 6;
    g->Matrix[1][2] = 6;
    g->Matrix[1][3] = 1;
    g->Matrix[2][3] = 2;
    g->Matrix[3][4] = 12;
    g->Matrix[4][5] = 1;
    g->Matrix[4][6] = 2;
    g->Matrix[5][6] = 2;
    g->Matrix[6][7] = 2;
    g->Matrix[6][8] = 9;
    g->Matrix[6][9] = 2;
    g->Matrix[6][10] = 2;
    g->Matrix[7][9] = 1;
    g->Matrix[8][10] = 9;
    g->Matrix[9][11] = 1;
    g->Matrix[10][11] = 2;
    g->Matrix[11][10] = 2;
    g->Matrix[11][12] = 5;
}

void dijkstra_method(MGraph* g, int dist[MAXSIZE], int path[MAXSIZE], int v0) {
    bool visited[MAXSIZE] = { false };
    for (int i = 0; i < g->VexNum; ++i) {
        dist[i] = g->Matrix[v0][i];
        if (dist[i] < INF) {
            path[i] = v0;
        }
        else {
            path[i] = -1;
        }
    }
    visited[v0] = true;
    dist[v0] = 0;
    for (int i = 1; i < g->VexNum; ++i) {
        int min_dist = INF;
        int u = -1;
        for (int j = 0; j < g->VexNum; ++j) {
            if (!visited[j] && dist[j] < min_dist) {
                u = j;
                min_dist = dist[j];
            }
        }
        if (u == -1) break;
        visited[u] = true;
        for (int j = 0; j < g->VexNum; ++j) {
            if (!visited[j] && g->Matrix[u][j] < INF) {
                int new_dist = dist[u] + g->Matrix[u][j];
                if (new_dist < dist[j]) {
                    dist[j] = new_dist;
                    path[j] = u;
                }
            }
        }
    }
}

void print_shortest_path(MGraph* g, int dist[MAXSIZE], int path[MAXSIZE], int start, int end) {
    if (dist[end] == INF) {
        cout << "无法到达目标景点" << endl;
        return;
    }
    cout << "最短路径长度为: " << dist[end] << endl;
    cout << "路径为: ";
    int current = end;
    string path_str = g->Vertex[current];
    while (current != start) {
        current = path[current];
        path_str = g->Vertex[current] + " -> " + path_str;
    }
    cout << path_str << endl;
}

int select(MGraph* g, string s) {
    for (int i = 0; i < g->VexNum; i++) {
        if (g->Vertex[i] == s) {
            return i;
        }
    }
    return -1;
}

int main() {
    MGraph g;
    InitMGraph(&g);
    string start, end;
    cout << "请输入起点景点名：" << endl;
    cin >> start;
    cout << "请输入终点景点名：" << endl;
    cin >> end;
    int vi = select(&g, start);
    int vj = select(&g, end);
    if (vi == -1 || vj == -1) {
        cout << "输入的景点名有误" << endl;
        return 0;
    }
    int dist[MAXSIZE];
    int path[MAXSIZE];
    dijkstra_method(&g, dist, path, vi);
    print_shortest_path(&g, dist, path, vi, vj);
    return 0;
}
