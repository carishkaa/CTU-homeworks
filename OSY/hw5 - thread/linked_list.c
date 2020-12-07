typedef struct node{
    struct node *next;
    int x;
    char *text;
} node_t;

typedef struct{
    int size;
    node_t *head;
    node_t *tail;
} list_t;

/* creates a new linked list */
list_t* create_list(void){
    list_t* new_list = (list_t*)malloc(sizeof(list_t));
    new_list->size = 0;
    new_list->head = NULL;
    new_list->tail = NULL;
    return (new_list);
}

/* adds data to the end of list */
void push(list_t* list, int x, char *text){
    node_t* node = malloc(sizeof(node_t));
    node->x = x;
    node->text = text;
    node->next = NULL;

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
node_t* pop(list_t* list){
    node_t* node = malloc(sizeof(node_t));

    if (list->size == 0){
        return NULL;
    }

    if (list->head){
        list->size--;
        node = list->head;
        list->head = list->head->next;
        if (list->size == 0)
            list->tail = NULL;
        return node;
    }
    else{
        return NULL;
    }
}

/*
 * returns elemets number
 */
int size(list_t* list){
    return list->size;
}