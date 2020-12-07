#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bst.h"

node_t* consolidation(int* nodes_array, int start, int end, int depth){
    if(start == end){
        node_t *node = new_node(nodes_array[start]);
        return node;
    }
    int size = end - start + 1;
    int max_nodes = pow(2, depth + 1) - 1; // full tree
    int missing_nodes = max_nodes - size;
    int max_subtree_diff = pow(2, depth - 1); // серединка последнего уровня

    int right_count, left_count;
    int right_depth, left_depth;

    if (missing_nodes >= max_subtree_diff){ // если а не попадает в правый стром
        right_count = pow(2, depth - 1) - 1;
        left_count = size - right_count - 1;
        right_depth = depth - 2;
    } else {
        left_count = pow(2, depth) - 1;
        right_count = size - left_count;
        right_depth = depth - 1;
    }
    left_depth = depth - 1;

    int index = start + left_count;
    node_t *node = new_node(nodes_array[index]);
    node->left = (left_count) ? consolidation(nodes_array, start, index - 1, left_depth) : NULL;
    node->right = (right_count) ? consolidation(nodes_array, index + 1, end, right_depth) : NULL;
    return node;
}

int consolidation_count = 0;
int cur_nodes_count = 1;
int depth = 0;

int main(){
    int n;
    scanf("%d", &n);

    // root
    node_t* root = NULL;
    int cur_key;
    scanf("%d", &cur_key);
    root = new_node(cur_key);

    // operations
    for (int i = 1; i < n; i++) {

        scanf("%d", &cur_key);
        _Bool duplicate_key_flag = 0;

        // insert
        if (cur_key > 0){
            int new_depth = -1;
            root = insert(root, cur_key, &duplicate_key_flag, &new_depth);
            if (!duplicate_key_flag) cur_nodes_count++;
            if (depth < new_depth) depth = new_depth;
        }
        // delete
        else {
            root = delete(root, (-1) * cur_key, &duplicate_key_flag);
            if (!duplicate_key_flag) cur_nodes_count--;
            depth = max_depth(root);
        }

        // consolidation
        if (pow(2, depth - 1) > cur_nodes_count){
            consolidation_count++;
            int* nodes_array = (int*) malloc (cur_nodes_count * sizeof(int));
            int ind = 0;
            store_in_order(root, nodes_array, &ind);
            delete_tree(root);

            depth = ceil(log2(cur_nodes_count + 1)) - 1;
            root = consolidation(nodes_array, 0, cur_nodes_count - 1, depth);


            free(nodes_array);
        }
    }

    printf("%d %d\n", consolidation_count, depth);
    return 0;
}