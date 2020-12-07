#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h> 
#include <pthread.h>
#include <string.h>
#include <semaphore.h>

typedef struct entry{
	int n_prints;
    char* word;
	struct entry* next;
}entry_t;

typedef struct{
	int count;
	entry_t* start;
	entry_t* end;
}linked_list_queue_t;

sem_t sem;
pthread_mutex_t lock_print;
pthread_mutex_t lock_pop;
pthread_mutex_t lock_check;

int check_status = 1;

linked_list_queue_t* list = NULL;

_Bool push(entry_t* entry){
	if(list->count == 0){
		list->start = (entry_t*)malloc(sizeof(entry_t));
		list->start->n_prints = entry->n_prints;
		list->start->word = entry->word;
		list->start->next = NULL;
		list->end = list->start;
	}
	else{
		entry_t* temp = (entry_t*)malloc(sizeof(entry_t));
		temp->n_prints = entry->n_prints;
		temp->word = entry->word;
		temp->next = NULL;
		list->end->next = temp;
		list->end = temp;
	}
	list->count++;
	return 1;
}
	
entry_t* pop(){
	if(list->count == 0)
		return NULL;
	else{
		entry_t* res = list->start;
		list->count--;
		list->start = list->start->next;
		return res;
	}
}
	

void clear(){
	entry_t* temp = list->start;
	entry_t* temp0 = NULL;
	while(temp != NULL){
		temp0 = temp;
		temp = temp->next;
		free(temp0->word);
		free(temp0);
	}
	list->start = NULL;
	list->end = NULL;
	list->count = 0;
}

void* producer()
{
    int ret, x;
    char *text;
    //printf("gg0\n");
    while ((ret = scanf("%d %ms", &x, &text)) == 2) {
        entry_t* new_entry = (entry_t*)malloc(sizeof(entry_t));
        new_entry->n_prints = x;
        new_entry->word = text;
        pthread_mutex_lock(&lock_pop);
        push(new_entry);
        pthread_mutex_unlock(&lock_pop);
        sem_post(&sem);
    }
    
    return NULL;

}

void* consumer(void* i_thread)
{
    //
    while(){
        pthread_mutex_lock(&lock_check);
        if(check_status == 0 && list->count == 0){
            return NULL;
        }
        printf("%d\n", check_status);
        pthread_mutex_unlock(&lock_check);
        sem_wait(&sem);
        //printf("gg2\n");
        
        entry_t* pop_entry = (entry_t*)malloc(sizeof(entry_t));
        pthread_mutex_lock(&lock_pop);
        pop_entry = pop(list);
        pthread_mutex_unlock(&lock_pop);
        pthread_mutex_lock(&lock_print);
        printf("Thread %d:", *((int*)i_thread));
        for(int i = 0; i < pop_entry->n_prints; i++){
            printf(" %s", pop_entry->word);
        }
        printf("\n");
        pthread_mutex_unlock(&lock_print);
        free(pop_entry->word);
        free(pop_entry);
    }
    printf("gg\n");
    return NULL;
} 


int main(int argc, char *argv[])
{
    
    pthread_mutex_init(&lock_print, NULL);
    pthread_mutex_init(&lock_pop, NULL);
    pthread_mutex_init(&lock_check, NULL);
    int n_consumers = 1;

    if(argc > 1){
        int tmp = atoi(argv[1]);
        if(tmp >= 1 && tmp <= sysconf(_SC_NPROCESSORS_ONLN)){
            n_consumers = tmp;
        }
        else{
            return 1;
        }
    }
    sem_init(&sem, 0, 0);
    
    list = (linked_list_queue_t*)malloc(sizeof(linked_list_queue_t));
    list->count = 0;

    pthread_t prod_id;
    pthread_attr_t prod_attr;

    pthread_attr_init(&prod_attr);

    pthread_create(&prod_id,&prod_attr,producer, NULL);

    pthread_t* cons_id = (pthread_t*)malloc(sizeof(pthread_t) * n_consumers);
    pthread_attr_t cons_attr;

    pthread_attr_init(&cons_attr);
    int* i_thread = (int*)malloc(sizeof(int) * n_consumers);
    for(int i = 0; i < n_consumers; i++){
        i_thread[i] = i+1;
        pthread_create(&cons_id[i],&cons_attr,consumer, &i_thread[i]);
    }
    
    pthread_join(prod_id, NULL);
    printf("gg1\n");
    printf("%d\n", n_consumers);
    pthread_mutex_lock(&lock_check);
    check_status = 0;
    pthread_mutex_unlock(&lock_check);
    sem_destroy(&sem);
    for(int i = 0; i < n_consumers; i++){
        pthread_join(cons_id[i], NULL);
    }
    
    pthread_mutex_destroy(&lock_print);
    pthread_mutex_destroy(&lock_pop);
    pthread_mutex_destroy(&lock_check);
    free(cons_id);
    free(i_thread);
    free(list);
    return 0;
}