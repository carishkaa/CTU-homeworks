#include <stdbool.h>
#include <stdlib.h>
#include "linked_list.h"

/* creates a new linked list */
list_t* create_list(void){
    list_t* new_list = (list_t*)malloc(sizeof(list_t));
    new_list->size = 0;
    new_list->head = NULL;
    new_list->tail = NULL;
    return (new_list);
}

/* adds data to the end of list */
void push(list_t* list, node_t* data){
    node_t* node = data;

    if (list->tail == NULL)
        list->head = node;
    else
        list->tail->next = node;
    list->tail = node;
    list->size++;
}

/*
 * gets the first element from the list and removes it from the list
 * returns: the first element on success; NULL otherwise
 */
bool pop(list_t* list){
    node_t* node;

    if (list->size == 0)
        return false;

    if (list->head){
        list->size--;
        node = list->head;
        list->head = list->head->next;
        if (list->size == 0)
            list->tail = NULL;
        free(node);
        return true;
    }
    return false;
}

node_t* front(list_t* list){
    if (list->size > 0)
        return list->head;
    return NULL;
}

/* returns elements number */
int size(list_t* list){
    return list->size;
}

bool is_empty(list_t* list){
    return (list->size > 0) ? false : true;
}