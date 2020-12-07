#include <iostream>
#include <vector>//
#include <queue>
#include <climits>

using namespace std;

// for grid 
#define PREKAZKA -1
#define EMPTY_CELL 0

struct Point{
    int x;
    int y;
};

struct Cell{
    Point coordinates;
    int type;
    int cur_path_time;
    int cur_znacka;
};

//Cell **grid;
vector< vector<Cell> > grid;//
queue<Point> q;
Point start;
int lines_number, columns_number, prekazka_number;
int types_number;

bool is_out_of_bonds(int x, int y){
    if (x < 1 || y < 1 || x > lines_number || y > columns_number)
        return true;
    if (grid[x][y].type == PREKAZKA)
        return true;
    return false;
}

void try_to_push(int x, int y, int prev_time, int znacka){
    if (!is_out_of_bonds(x, y) && (grid[x][y].cur_znacka < znacka)){
        Point point = {x, y};
        grid[x][y].cur_znacka = znacka;
        grid[x][y].cur_path_time = prev_time + 1;
        q.push(point);
    }
}

int bfs(){
    q.push(start);
    while (!q.empty()){
        Point curr = q.front();
        q.pop();
        int curr_time = grid[curr.x][curr.y].cur_path_time;

        // pokud je to posledni znacka tak porovname ji s best_min_time
        if (grid[curr.x][curr.y].type == types_number && grid[curr.x][curr.y].cur_znacka == types_number - 1){
            return curr_time;
        }
        
        // sebrat znacku
        int new_znacka = grid[curr.x][curr.y].cur_znacka;
        if (new_znacka == grid[curr.x][curr.y].type - 1)
            new_znacka++;

        try_to_push(curr.x, curr.y + 1, curr_time, new_znacka);
        try_to_push(curr.x, curr.y - 1, curr_time, new_znacka);
        try_to_push(curr.x + 1, curr.y, curr_time, new_znacka);
        try_to_push(curr.x - 1, curr.y, curr_time, new_znacka);

    }
    return 0;
}



int main(){
    cin >> lines_number >> columns_number >> prekazka_number >> types_number;

    int r0, s0;
    cin >> r0 >> s0;
    start.x = r0;
    start.y = s0;

    // array initialization
    grid = vector< vector<Cell> >(lines_number + 1, vector<Cell>(columns_number + 1));
    for (int i = 1; i <= lines_number; i++){
        for (int j = 1; j <= columns_number; j++){
            Point point = {i, j};
            grid[i][j].coordinates = point;
            grid[i][j].cur_znacka = -1;
            grid[i][j].cur_path_time = 0;
            grid[i][j].type = EMPTY_CELL;
        }
    }
    grid[start.x][start.y].cur_znacka = 0;

    for (int i = 0; i < prekazka_number; i++){
        int x, y;
        cin >> x >> y;
        grid[x][y].type = PREKAZKA;
    }

    for (int i = 1; i <= types_number; i++){
        int count;
        cin >> count;
        for (int j = 0; j < count; j++){
            int x, y;
            cin >> x >> y;
            grid[x][y].type = i;
        }
    }

//     cout << endl;
//     for (int j = 1; j <= columns_number; ++j) {
//        for (int i = 1; i <= lines_number; ++i) {
//            printf("%2d ", grid[i][j].type);
//        }
//        printf("\n");
//    }

    int result = bfs();
    cout << result << endl;

    return 0;
}
