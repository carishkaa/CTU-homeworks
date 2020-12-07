#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include "linked_list.c"

sem_t semaphore;
pthread_mutex_t mutex_cons;
pthread_mutex_t mutex_list;
pthread_mutex_t mutex_prod_status;

/* 
 *  prod_status = 0 - producer runs;
 *  prod_status > 0 - producer ends: bad stdin; 
 *  prod_status < 0 - producer ends: stdin was closed; 
 */
int prod_status = 0;

list_t *list = NULL;

void *prod(void *arg){
    int ret, x;
    char *text;
    
    while ((ret = scanf("%d %ms", &x, &text)) == 2){
        if (x < 0){
            ret = 1;
            break;
        }
        pthread_mutex_lock(&mutex_list);
        push(list, x, text);
        pthread_mutex_unlock(&mutex_list);
        sem_post(&semaphore);
    }

    pthread_mutex_lock(&mutex_prod_status);
    if (ret >= 0){  //bad data
        fprintf(stderr, "Error: producer sent bad data\n");
        prod_status = 1;
    } else {     // stdin was closed (ctrl+D)
        prod_status = -1; 
    }
    pthread_mutex_unlock(&mutex_prod_status);
    
    return NULL;
}

void *cons(void *arg){
    int n = (intptr_t)arg;
    while(1){
        sem_wait(&semaphore);

        pthread_mutex_lock(&mutex_prod_status);
        int tmp_prod_status = prod_status;
        pthread_mutex_unlock(&mutex_prod_status);

        //pop
        pthread_mutex_lock(&mutex_list);
        if ((tmp_prod_status != 0) && size(list) == 0){     // pokud producer uz nebezi a list je prazdny:
            fprintf(stderr, "cons %d end\n", n);            // ukoncime consumer
            pthread_mutex_unlock(&mutex_list);
            break;
        }
        node_t* node = pop(list);
        pthread_mutex_unlock(&mutex_list);

        // write 
        pthread_mutex_lock(&mutex_cons);        
        printf("Thread %d:", n);
        for (int i = 0; i < node->x; i++)
            printf(" %s", node->text);
        printf("\n");
        free(node->text);
        free(node);
        pthread_mutex_unlock(&mutex_cons);
        
    }
    return NULL;
}


int main(int argc, char *argv[]){
    
    int N = (argc == 1) ? 1 : atoi(argv[1]);
    if (N < 1 || N > sysconf(_SC_NPROCESSORS_ONLN)){
        fprintf(stderr, "Error: bad argument N\n");
        exit(1);
    }

    pthread_t producer_thread;
    pthread_t consumer_threads[N];
    
    list = create_list();

    sem_init(&semaphore, 0, 0);
    pthread_mutex_init(&mutex_cons, NULL);
    pthread_mutex_init(&mutex_list, NULL);
    pthread_mutex_init(&mutex_prod_status, NULL);

    pthread_create(&producer_thread, NULL, &prod, NULL);
    for (int thread_index = 1; thread_index <= N; thread_index++)
        pthread_create(&consumer_threads[thread_index], NULL, &cons, (void *)(intptr_t)thread_index);

    pthread_join(producer_thread, NULL);

    for (int thread_index = 1; thread_index <= N; thread_index++){
        sem_post(&semaphore);
    }
    for (int thread_index = 1; thread_index <= N; thread_index++){
        fprintf(stderr, "Wait for a cons %d\n", thread_index);
        pthread_join(consumer_threads[thread_index], NULL);
    }
    
    sem_destroy(&semaphore);
    pthread_mutex_destroy(&mutex_cons);
    pthread_mutex_destroy(&mutex_list);
    pthread_mutex_destroy(&mutex_prod_status);
    
    free(list);

    prod_status = (prod_status == -1) ? 0 : prod_status;
    exit(prod_status);
}
