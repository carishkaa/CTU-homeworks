#include <iostream>
#include <queue>
#include <climits>

using namespace std;

// for containers array
#define CONTAINER -1
#define SLOW_STEP 1
#define FAST_STEP 0

struct Depot{
    int width;
    int height;
    int **containers;
    int **time_layout;
}depot;

struct Point{
    int x;
    int y;
};

int fast_step, slow_step, containers_number;
int container_width, container_height;

bool is_out_of_bonds(int x, int y){
    if (x < 0 || y < 0 || x >= depot.width || y >= depot.height)
        return true;
    return false;
}

bool is_around_container(int x, int y, int x1, int y1, int x2, int y2){
    if (x == x1 - 1 || x == x2 + 1 || y == y1 - 1 || y == y2 + 1)
        return true;
    return false;
}

bool intersection(int x, int y, int step){
    for (int i = x; i < x + container_width; i++)
        for (int j = y; j < y + container_height; j++) 
            if (depot.containers[i][j] == step)
                return true;
    return false;
}

Point original, target;
queue<Point> q;

bool fits(int x, int y){
    if (is_out_of_bonds(x, y))
        return false;
    if (is_out_of_bonds(x + container_width - 1, y + container_height - 1))
        return false;
    if (intersection(x, y, CONTAINER))
        return false;
    return true;
}

void push_if_fits(int x, int y, int prev_is_slow, int prev_time){
    if (fits(x, y)){
        int step = (prev_is_slow || intersection(x, y, SLOW_STEP)) ? slow_step : fast_step;
        if (depot.time_layout[x][y] == 0 || depot.time_layout[x][y] > prev_time + step){
            depot.time_layout[x][y] = prev_time + step;
            Point point = {x, y};
            q.push(point);
        }
    }
}

int bfs(){
    int best_min_time = INT_MAX;

    Point start = {original.x, original.y};
    q.push(start);
    while (!q.empty()){
        Point curr = q.front();
        q.pop();

        int curr_time = depot.time_layout[curr.x][curr.y];
        if (curr_time >= best_min_time)
            continue;

        if (curr.x == target.x && curr.y == target.y && curr_time < best_min_time)
            best_min_time = curr_time;

        int is_slow = intersection(curr.x, curr.y, SLOW_STEP);
        push_if_fits(curr.x, curr.y + 1, is_slow, curr_time);
        push_if_fits(curr.x, curr.y - 1, is_slow, curr_time);
        push_if_fits(curr.x + 1, curr.y, is_slow, curr_time);
        push_if_fits(curr.x - 1, curr.y, is_slow, curr_time);
    }
    return (best_min_time != INT_MAX) ? best_min_time : -1;
}


int main(){
    // input
    cin >> depot.width >> depot.height;
    cin >> fast_step >> slow_step >> containers_number;

    // arrays initialization
    depot.containers = new int*[depot.width];
    depot.time_layout = new int*[depot.width];
    for (int i = 0; i < depot.width; i++){
        depot.containers[i] = new int[depot.height];
        depot.time_layout[i] = new int[depot.height];
        for (int j = 0; j < depot.height; j++){
            depot.time_layout[i][j] = 0;
            depot.containers[i][j] = (i == 0 || j == 0 || i == depot.width - 1 || j == depot.height -1) ? SLOW_STEP : FAST_STEP;
        }
    }

    // containers
    for (int i = 0; i < containers_number; i++){
        int x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;

        if (i == 0){
            original.x = x1;
            original.y = y1;
            container_width = x2 - x1 + 1;
            container_height = y2 - y1 + 1;
            continue;
        }

        for (int x = x1 - 1; x <= x2 + 1; x++){
            for (int y = y1 - 1; y <= y2 + 1; y++){
                if (is_out_of_bonds(x, y)) continue;
                if (is_around_container(x, y, x1, y1, x2, y2) && depot.containers[x][y] != CONTAINER)
                    depot.containers[x][y] = SLOW_STEP; // near the container
                else 
                    depot.containers[x][y] = CONTAINER; // container
            }
        }
    }

    // target position
    cin >> target.x >> target.y;

    // bfs
    int result = bfs();

    cout << result << endl;
    
    return 0;
}