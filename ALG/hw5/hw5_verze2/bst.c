#include <stdlib.h>
#include <math.h>
#include "bst.h"

node_t *new_node(int k){
    node_t *new_node = (node_t*)malloc(sizeof(node_t));
    new_node->key = k;
    new_node->left = new_node->right = NULL;
    return new_node;
}

node_t* find_max(node_t *node){
    while (node->right != NULL)
        node = node->right;
    return node;
}

int max_depth(node_t* node){
    if(node == NULL || (node->left == NULL && node->right == NULL))
        return 0;
    int l = max_depth(node->left);
    int r = max_depth(node->right);
    return (1 + ((l > r) ? l : r));
}

void store_in_order(node_t* node, int* nodes_array, int *index) {
    if (node == NULL)
        return;
    store_in_order(node->left, nodes_array, index);
    nodes_array[(*index)++] = node->key;
    store_in_order(node->right, nodes_array, index);
}

void delete_tree(node_t *node){
    if (node == NULL)
        return;
    delete_tree(node->left);
    delete_tree(node->right);
    free(node);
}

node_t* insert(node_t* node, int key, _Bool *duplicate_flag, int *depth){
    (*depth)++;
    // if tree is empty
    if (node == NULL) {
        return new_node(key);
    }

    // can't insert a duplicate
    if (key == node->key) {
        *duplicate_flag = 1;
        return node;
    }

    // chose direction
    if (key < node->key)
        node->left = insert(node->left, key, duplicate_flag, depth);
    else
        node->right = insert(node->right, key, duplicate_flag, depth);

    return node;
}



node_t* delete(node_t *node, int key, _Bool *duplicate_flag){
    *duplicate_flag = 1;
    if (node == NULL)
        return NULL;
    else if (key < node->key)
        node->left = delete(node->left, key, duplicate_flag);
    else if (key > node->key)
        node->right = delete(node->right, key, duplicate_flag);
    else {
        *duplicate_flag = 0;
        // Case 1: No Child
        if (node->left == NULL && node->right == NULL){
            free(node);
            node = NULL;
            // Case 2: one child
        } else if(node->left == NULL) {
            node_t *tmp = node;
            node = node->right;
            free(tmp);
        } else if(node->right == NULL) {
            node_t *tmp = node;
            node = node->left;
            free(tmp);
            // Case 3: two child
        } else {
            node_t *tmp = find_max(node->left);
            node->key = tmp->key;
            node->left = delete(node->left, tmp->key, duplicate_flag);
        }
    }
    return node;
}