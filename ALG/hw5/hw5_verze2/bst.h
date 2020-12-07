#ifndef BST_H
#define BST_H

typedef struct Node{
    int key;
    struct Node *left;
    struct Node *right;
} node_t;

node_t *new_node(int k);

node_t* find_max(node_t *node);

int max_depth(node_t* node);

void store_in_order(node_t* node, int* nodes_array, int *index);

void delete_tree(node_t *node);

node_t* insert(node_t* node, int key, _Bool *duplicate_flag, int *depth);

node_t* delete(node_t *node, int key, _Bool *duplicate_flag);

#endif // BST_H