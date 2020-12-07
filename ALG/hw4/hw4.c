#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "linked_list.c"

int lines_count, column_count;
int E_init, D_mov, D_env;
cell_t **grid;
list_t *queue;

int rules(int I1, int I2, int old_engine_stage){
    int new_engine_stage;
    if (I1 < I2)
        new_engine_stage = old_engine_stage + D_mov;
    else if (I1 == I2)
        new_engine_stage = old_engine_stage;
    else if (I1 > I2 && old_engine_stage >= D_mov)
        new_engine_stage = old_engine_stage - D_mov;
    else
        new_engine_stage = 0;

    if (new_engine_stage >= D_env + I2)
        new_engine_stage -= D_env;
    else if (new_engine_stage > I2)
        new_engine_stage = I2;
    return new_engine_stage;
}


void calculate_and_push(int line, int column, node_t* prev_node){
    node_t* neigh_node = (node_t*)malloc(sizeof(node_t));
    neigh_node->cell = &grid[line][column];
    neigh_node->distance = prev_node->distance + 1;
    neigh_node->next = NULL;

    int last_engine_stage = neigh_node->cell->engine_ion_stage; //ion stage (need for situation when node is not fresh)
    int new_engine_stage = rules(prev_node->cell->cell_ion_stage, neigh_node->cell->cell_ion_stage, prev_node->cell->engine_ion_stage);
    if (new_engine_stage > last_engine_stage || (neigh_node->cell->line == lines_count - 1 && neigh_node->cell->column == column_count - 1)){
        neigh_node->cell->engine_ion_stage = new_engine_stage;
        push(queue, neigh_node);
    }
}


int bfs(node_t* start){
    queue = create_list();
    push(queue, start);

    node_t* cur_node;
    do {
        cur_node = front(queue);

        // end
        if (cur_node->cell->line == lines_count - 1 && cur_node->cell->column == column_count - 1)
            return cur_node->distance;

        // nahoru
        if (cur_node->cell->line > 0)
            calculate_and_push(cur_node->cell->line - 1, cur_node->cell->column, cur_node);
        // dolu
        if (cur_node->cell->line < lines_count - 1)
            calculate_and_push(cur_node->cell->line + 1, cur_node->cell->column, cur_node);
        // nalevo
        if (cur_node->cell->column > 0)
            calculate_and_push(cur_node->cell->line, cur_node->cell->column - 1, cur_node);
        // napravo
        if (cur_node->cell->column < column_count - 1)
            calculate_and_push(cur_node->cell->line, cur_node->cell->column + 1, cur_node);

        pop(queue);
    } while (!is_empty(queue));

    return -1;
}


int main(){

    //input
    scanf("%d %d %d %d %d", &lines_count, &column_count, &E_init, &D_mov, &D_env);

    grid = (cell_t**)malloc(lines_count * sizeof(cell_t*));
    for (int i = 0; i < lines_count; i++){
        grid[i] = (cell_t*)malloc(column_count * sizeof(cell_t));
        for (int j = 0; j < column_count; j++){
            scanf("%d", &grid[i][j].cell_ion_stage);
            grid[i][j].engine_ion_stage = 0;
            grid[i][j].line = i;
            grid[i][j].column = j;
        }
    }

    // bfs
    node_t* start = (node_t*)malloc(sizeof(node_t));

    grid[0][0].engine_ion_stage = E_init;
    start->cell = &grid[0][0];
    start->distance = 0;
    start->next = NULL;

    int dist = bfs(start);
    printf("%d\n", dist);

    // deallocate
    for (int i = 0; i < lines_count; i++)
        free(grid[i]);
    free(grid);

    return 0;
}
