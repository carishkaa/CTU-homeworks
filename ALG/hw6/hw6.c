#include <stdio.h>
#include <stdlib.h>

typedef struct connection{
    int cost;
    int destination;
}connection_t;

typedef struct city{
    int idx;
    int route_here;
    connection_t *connections;
    int connection_count;
    int is_zero_median;
    int is_in_queue;
    //int start_point;
    //int end_point;
}city_t;

typedef struct route{
    int start;
    int end;
    int cost;
}route_t;

typedef struct entry {
    int *value;
    struct entry *next;
} queue_entry_t;

typedef struct {
    queue_entry_t *head;
    queue_entry_t *end;
}queue_t;

int queue_push(int value, queue_t *queue);
int* queue_pop(queue_t *queue);
void queue_init(queue_t **queue);
int queue_is_empty(queue_t *queue);

route_t bfs_route(city_t *cities, int start);

int main() {
    int city_count, connection_count, zero_median_count, routes_count;
    scanf("%d %d %d %d", &city_count, &connection_count, &zero_median_count, &routes_count);

    city_t *cities = (city_t*)malloc(city_count * sizeof(city_t));
    int *zero_medians = (int*)malloc(zero_median_count * sizeof(int));
    route_t *route = (route_t*)malloc(zero_median_count* sizeof(route_t));

    for(int i = 0 ; i < city_count; ++i){
        cities[i].idx = i;
        cities[i].route_here = 0;
        cities[i].connections = (connection_t*)malloc(city_count * sizeof(connection_t));
        cities[i].connection_count = 0;
        cities[i].is_zero_median = 0;
        cities[i].is_in_queue = 0;
        //cities[i].start_point = -1;
    }

    int read_tmp;
    for(int i = 0; i < zero_median_count; ++i){
        scanf("%d", &read_tmp);
        read_tmp--;
        cities[read_tmp].is_zero_median = 1;
        zero_medians[i] = read_tmp;
    }

    int start_tmp, end_tmp;
    for(int i = 0; i < connection_count; ++i){
        scanf("%d %d %d", &start_tmp, &end_tmp, &read_tmp);
        start_tmp--; end_tmp--;
        cities[start_tmp].connections[cities[start_tmp].connection_count].destination = end_tmp;
        cities[start_tmp].connections[cities[start_tmp].connection_count++].cost = read_tmp;
    }

    for(int i = 0; i < zero_median_count; ++i)
    {
        route[i] = bfs_route(cities, zero_medians[i]);
        if(i == zero_median_count - 1)
            break;
        for(int j = 0; j < city_count; ++j){
            cities[j].is_in_queue = 0;
            cities[j].route_here = 0;
        }
    }

    int curr_min = 0, global_min = 2000000000;
    for(int j = 0; j < zero_median_count; ++j){
        //printf("yy");
        route_t curr_rout = route[j];
        for(int i = 0; i < routes_count; ++i){
            curr_min += curr_rout.cost;
            if(i == routes_count - 1)
                break;
            for(int k = 0; k < zero_median_count; ++k){
                if(route[k].start == curr_rout.end){
                    curr_rout = route[k];
                    break;
                }
            }
        }
        if(curr_min < global_min)
            global_min = curr_min;
        curr_min = 0;
    }

    printf("%d\n", global_min);

    return 0;
}

route_t bfs_route(city_t *cities, int start){
    queue_t* queue;
    queue_init(&queue);
    cities[start].is_in_queue = 1;
    //cities[start].start_point = start;
    queue_push(start, queue);

    route_t min_route;
    min_route.start = start;
    min_route.cost = 2000000000;

    while(!queue_is_empty(queue)){
        int current = *queue_pop(queue);
        cities[current].is_in_queue = 0;
        if(cities[current].route_here != 0 && cities[current].is_zero_median){
            if( cities[current].route_here < min_route.cost){
                min_route.end = current;
                min_route.cost = cities[current].route_here;
            }
            continue;
        }

        for(int i = 0; i < cities[current].connection_count; ++i){
            if(cities[cities[current].connections[i].destination].route_here == 0
                || cities[cities[current].connections[i].destination].route_here
                        > cities[current].route_here + cities[current].connections[i].cost){
                cities[cities[current].connections[i].destination].route_here = cities[current].route_here + cities[current].connections[i].cost;

                if(!cities[cities[current].connections[i].destination].is_in_queue){
                    cities[cities[current].connections[i].destination].is_in_queue = 1;
                    queue_push(cities[current].connections[i].destination, queue);
                }
            }
        }
    }

    return min_route;
}

int queue_push(int value, queue_t *queue)
{
    int ret = 0;
    queue_entry_t *new_entry = (queue_entry_t*)malloc(sizeof(queue_entry_t));
    new_entry->value = (int*)malloc(sizeof(int));
    if (new_entry) {
        *(new_entry->value) = value;
        new_entry->next = NULL;
        if (queue->end)
            queue->end->next = new_entry;
        else
            queue->head = new_entry;
        queue->end = new_entry;
    } else
        ret = 1;
    return ret;
}

int* queue_pop(queue_t *queue)
{
    void *ret = NULL;
    if (queue->head) {
        ret = queue->head->value;
        queue_entry_t *tmp = queue->head;
        queue->head = queue->head->next;
        free(tmp);
        if (queue->head == NULL)
            queue->end = NULL;
    }
    return ret;
}

void queue_init(queue_t **queue) {
    *queue = (queue_t*)malloc( sizeof(queue_t));
    (*queue)->head = NULL;
    (*queue)->end = NULL;
}

int queue_is_empty(queue_t *queue){
    if (queue->head == NULL)
        return 1;
    return 0;
}