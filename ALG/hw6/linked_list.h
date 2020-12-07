#ifndef LINKED_LIST_H
#define LINKED_LIST_H

typedef struct{
    int dest_village;
    int ride_time;
} neighbour_t;

typedef struct{
    int index;
    int tourist_index;
    int visit_time;
    int neighbours_count;
    neighbour_t* neighbours;
    int path_count;
    int *path;
    int path_tourist_index;
    int path_time;
} village_t;

typedef struct node{
    village_t* village;
    struct node *next;
} node_t;

typedef struct{
    int size;
    node_t *head;
    node_t *tail;
} list_t;

/* creates a new linked list */
list_t* create_list(void);

/* adds data to the end of list */
void push(list_t* list, node_t* data);

/*
 * gets the first element from the list and removes it from the list
 * returns: the first element on success; NULL otherwise
 */
bool pop(list_t* list);

/* returns the first element from the list */
node_t* front(list_t* list);

/* returns elements number */
int size(list_t* list);

bool is_empty(list_t* list);

#endif // LINKED_LIST_H
