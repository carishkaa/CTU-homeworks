#include <stdio.h>
#include <stdlib.h>

#define NOONE 0
#define T1 1
#define T2 2

typedef struct data {
    int idx;
    int TYPE;
    int neighbours_count;
    int *neighbours;
} node_t;

int vertex_count, edge_count, T1_count, T2_count;
int temp_T1_count, temp_T2_count;
node_t *nodes;
int best_result = 0;


int get_profit_T1(int idx) {
    int curr_profit = 0;
    for (int i = 0; i < nodes[idx].neighbours_count; i++) {
        int neighbour_idx = nodes[idx].neighbours[i];
        if (nodes[neighbour_idx].TYPE == T1 || nodes[neighbour_idx].TYPE == T2) {
            curr_profit++;
        }
    }
    // printf("type1, %d\n", curr_profit);
    return curr_profit;
}

int get_profit_T2(int idx) {
    int curr_profit = 0;
    for (int i = 0; i < nodes[idx].neighbours_count; i++) {
        int neighbour_idx = nodes[idx].neighbours[i];
        if (nodes[neighbour_idx].TYPE == NOONE) {
            curr_profit++;
        }
    }
    // printf("type2, %d\n", curr_profit);
    return curr_profit;
}

void check_combination() {

    int curr_profit = 0;

    for (int i = 0; i < vertex_count; i++) {
        if (nodes[i].TYPE == T1) {
            curr_profit += get_profit_T1(i);
        }
        if (nodes[i].TYPE == T2) {
            curr_profit += get_profit_T2(i);
        }
    }

    if (curr_profit > best_result) {
        best_result = curr_profit;
        printf("best: %d\n", best_result);
    }
}




void recursion_for_T2 (int start_idx) {

    if (temp_T2_count > 0) {
        for (int i = start_idx; i < vertex_count; i++) {
            if (nodes[i].TYPE == NOONE) {
                nodes[i].TYPE = T2;
                temp_T2_count--;
                recursion_for_T2(i + 1);
                nodes[i].TYPE = NOONE;
                temp_T2_count++;
            }
        }
    } else {
        // for (int i = 0; i < vertex_count; i++) {
        //     printf("%d ", nodes[i].TYPE);
        // }
        // printf("\n");
        check_combination();
    }
}



void recursion_for_T1(int start_idx) {
    if (temp_T1_count > 0) {
        for (int i = start_idx; i < vertex_count; i++) {
            nodes[i].TYPE = T1;
            temp_T1_count--;
            recursion_for_T1(i + 1);
            nodes[i].TYPE = NOONE;
            temp_T1_count++; 
        }
    } else {
        int start_for_T2;

        for (int i = 0; i < vertex_count; i++) {
            if (nodes[i].TYPE == T1 && nodes[i+1].TYPE == NOONE){
                start_for_T2 = i + 1;
                // printf("START T2 FROM: %d.\n", i+1);
                break;
            }
        }
        recursion_for_T2(start_for_T2);
    }
}



int main() {

    scanf("%d %d %d %d\n", &vertex_count, &edge_count, &T1_count, &T2_count);
    temp_T1_count = T1_count;
    temp_T2_count = T2_count;

    nodes = (node_t *)malloc(vertex_count * sizeof(node_t));

    for (int i = 0; i < vertex_count; i++) {
        nodes[i].idx = i;
        nodes[i].TYPE = NOONE;
        nodes[i].neighbours_count = 0;
        nodes[i].neighbours = (int *)calloc(1, sizeof(int));
    }

    for (int i = 0; i < edge_count; i++){
        int from, to;
        scanf("%d %d\n", &from, &to);
        from--;
        to--;

        int from_neigh_count = nodes[from].neighbours_count;
        nodes[from].neighbours[from_neigh_count++] = to;
        nodes[from].neighbours_count = from_neigh_count;
        nodes[from].neighbours = (int *)realloc(nodes[from].neighbours, (from_neigh_count + 1) * sizeof(int));

        int to_neigh_count = nodes[to].neighbours_count;
        nodes[to].neighbours[to_neigh_count++] = from;
        nodes[to].neighbours_count = to_neigh_count;
        nodes[to].neighbours = (int *)realloc(nodes[to].neighbours, (to_neigh_count + 1) * sizeof(int));
    }

    for (int i = 0; i < vertex_count; i++){
        nodes[i].TYPE = T1;
        temp_T1_count--;
        recursion_for_T1(i + 1);
        nodes[i].TYPE = NOONE;
        temp_T1_count++;
    }

    printf("%d\n", best_result);


     // test
     /*for (int i = 0; i < vertex_count; i++) {
         printf("idx: %d, neighbour count: %d.\n", nodes[i].idx, nodes[i].neighbours_count);
         for (int j = 0; j < nodes[i].neighbours_count; j++) {
             printf("%d ", nodes[i].neighbours[j]);
         }
         printf("\n");
     }*/


    return 0;
}
