#include <iostream>
#include <queue>
using namespace std;

#define MAX_SIZE 6

bool can[MAX_SIZE][MAX_SIZE] = { false };
char maze[MAX_SIZE][MAX_SIZE];
int pathMap[MAX_SIZE][MAX_SIZE] = { 0 };

typedef struct Point {
    int x, y;
    int step;
} Point;

void markPath(int endX, int endY) {
    int step = pathMap[endX][endY];
    int x = endX;
    int y = endY;

    while (step > 1) {
        maze[x][y] = '0' + (step % 10);

        if (x > 1 && pathMap[x - 1][y] == step - 1) x--;
        else if (x < MAX_SIZE - 1 && pathMap[x + 1][y] == step - 1) x++;
        else if (y > 1 && pathMap[x][y - 1] == step - 1) y--;
        else if (y < MAX_SIZE - 1 && pathMap[x][y + 1] == step - 1) y++;
        step--;
    }

    maze[1][1] = '1';
}

void printMaze(int size) {
    for (int i = 1; i <= size; ++i) {
        for (int j = 1; j <= size; ++j) {
            cout << maze[i][j] << " ";
        }
        cout << endl;
    }
}

int main() {
    int size = 4;
    queue<Point> q;

    for (int i = 1; i <= size; ++i) {
        for (int j = 1; j <= size; ++j) {
            int x;
            cin >> x;
            can[i][j] = (x == 0);
            maze[i][j] = (x == 0) ? '0' : '*';
        }
    }

    q.push({ 1, 1, 1 });
    can[1][1] = false;
    pathMap[1][1] = 1;

    while (!q.empty()) {
        Point current = q.front(); q.pop();

        if (current.x == size && current.y == size) {
            markPath(size, size);
            break;
        }

        int dx[4] = { 1, -1, 0, 0 };
        int dy[4] = { 0, 0, 1, -1 };
        for (int dir = 0; dir < 4; ++dir) {
            int nx = current.x + dx[dir];
            int ny = current.y + dy[dir];

            if (can[nx][ny]) {
                q.push({ nx, ny, current.step + 1 });
                can[nx][ny] = false;
                pathMap[nx][ny] = current.step + 1;
            }
        }
    }

    printMaze(size);

    return 0;
}
