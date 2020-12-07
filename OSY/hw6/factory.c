#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
 
typedef struct worker_t{
    char *name;
    int place;
    pthread_t thread;
    bool leave;
} worker_t;

int workers_count = 0;
worker_t *workers = NULL;

pthread_mutex_t mutex;
pthread_cond_t cond_worker;

bool exit_flag = false;
bool eof = false;

/* how many people are currently working*/ 
int cur_working_count = 0;

enum place {
    NUZKY, VRTACKA, OHYBACKA, SVARECKA, LAKOVNA, SROUBOVAK, FREZA,
    _PLACE_COUNT
};
 
const char *place_str[_PLACE_COUNT] = {
    [NUZKY] = "nuzky",
    [VRTACKA] = "vrtacka",
    [OHYBACKA] = "ohybacka",
    [SVARECKA] = "svarecka",
    [LAKOVNA] = "lakovna",
    [SROUBOVAK] = "sroubovak",
    [FREZA] = "freza",
};

const unsigned int sleep_time[_PLACE_COUNT] = {
    [NUZKY] = 100000,
    [VRTACKA] = 200000,
    [OHYBACKA] = 150000,
    [SVARECKA] = 300000,
    [LAKOVNA] = 400000,
    [SROUBOVAK] = 250000,
    [FREZA] = 500000
};
 
enum product {
    A, B, C,
    _PRODUCT_COUNT
};
 
const char *product_str[_PRODUCT_COUNT] = {
    [A] = "A",
    [B] = "B",
    [C] = "C",
};
 
int find_string_in_array(const char **array, int length, char *what){
    for (int i = 0; i < length; i++)
	if (strcmp(array[i], what) == 0)
	    return i;
    return -1;
}

int ready_places[_PLACE_COUNT] = { 0 };
int occupation_places[_PLACE_COUNT] = { 0 }; // how many people works on each place
 
#define _PHASE_COUNT 6
int parts[_PRODUCT_COUNT][_PHASE_COUNT] = { 0 };

const int parts_places[_PRODUCT_COUNT][_PHASE_COUNT] = { 
    { NUZKY,  VRTACKA,  OHYBACKA, SVARECKA,  VRTACKA, LAKOVNA }, 
    { VRTACKA,  NUZKY,   FREZA,  VRTACKA, LAKOVNA,  SROUBOVAK },
    { FREZA,  VRTACKA,  SROUBOVAK, VRTACKA, FREZA, LAKOVNA}
};
 

bool have_work(int *product, int *phase, int n){
    for (int phase_ind = _PHASE_COUNT - 1; phase_ind >= 0; phase_ind--)
        for (int product_ind = 0; product_ind < _PRODUCT_COUNT; product_ind++)
            if (parts_places[product_ind][phase_ind] == workers[n].place && parts[product_ind][phase_ind] > 0 && ready_places[workers[n].place] > 0){
                *phase = phase_ind;
                *product = product_ind;
                return true;
            }
    return false;
}

void exit_control(){
    for (int i = 0; i < workers_count; i++)
        for (int phase_ind = _PHASE_COUNT - 1; phase_ind >= 0; phase_ind--)
            for (int product_ind = 0; product_ind < _PRODUCT_COUNT; product_ind++)
                if (parts_places[product_ind][phase_ind] == workers[i].place && parts[product_ind][phase_ind] > 0 && ready_places[workers[i].place] > 0 && !workers[i].leave){
                    pthread_cond_broadcast(&cond_worker);
                    return;
                }

    if (cur_working_count == 0)
        exit_flag = true;
    return;
}

void *worker(void *arg){
    int n = (intptr_t)arg;
    while (1){
        pthread_mutex_lock(&mutex);

        //check if there is something for this worker
        int cur_phase = -1, cur_product = -1;

        if (have_work(&cur_product, &cur_phase, n)){

            //prepare for work
            cur_working_count++;
            parts[cur_product][cur_phase]--;
            ready_places[workers[n].place]--;
            pthread_mutex_unlock(&mutex);

            // working
            printf("%s %s %d %s\n",  workers[n].name, place_str[workers[n].place], cur_phase + 1, product_str[cur_product]);
            usleep(sleep_time[workers[n].place]);

            //after work
            pthread_mutex_lock(&mutex);
            ready_places[workers[n].place]++;
            if (cur_phase == _PHASE_COUNT - 1)
                printf("done %s\n", product_str[cur_product]);
            else 
                parts[cur_product][cur_phase + 1]++;
            cur_working_count--;
            pthread_cond_broadcast(&cond_worker);
        }
       
        // exit check
        if (workers[n].leave || exit_flag){
            pthread_mutex_unlock(&mutex);
            break;
        }

        //waiting for signal
        pthread_cond_wait(&cond_worker, &mutex);
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

int main(int argc, char **argv){

    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond_worker, NULL);
    
    while (1) {
        char *line, *cmd, *arg1, *arg2, *arg3, *saveptr;
        int s = scanf(" %m[^\n]", &line);
        if (s == EOF){
            eof = true;
            break;
        }

        if (s == 0)
            continue;
    
        cmd  = strtok_r(line, " ", &saveptr);
        arg1 = strtok_r(NULL, " ", &saveptr);
        arg2 = strtok_r(NULL, " ", &saveptr);
        arg3 = strtok_r(NULL, " ", &saveptr);
    

        //START
        if (strcmp(cmd, "start") == 0 && arg1 && arg2 && !arg3) {
            int place = find_string_in_array(place_str, _PLACE_COUNT, arg2);
            if (place >= 0){
                pthread_mutex_lock(&mutex);
                workers = (worker_t*)realloc(workers, (workers_count + 1) * sizeof(worker_t));    
                workers[workers_count].name = strdup(arg1);
                workers[workers_count].place = place;
                workers[workers_count].leave = false;
                occupation_places[place]++;
                pthread_create(&workers[workers_count].thread, NULL, &worker, (void *)(intptr_t)(workers_count));
                workers_count++;
                pthread_mutex_unlock(&mutex);
                pthread_cond_broadcast(&cond_worker);
            }

        // MAKE
        } else if (strcmp(cmd, "make") == 0 && arg1 && !arg2) {
            int product = find_string_in_array(product_str, _PRODUCT_COUNT, arg1); // 0 = A, 1 = B, 2 = C
            if (product >= 0){
                pthread_mutex_lock(&mutex);
                parts[product][0]++;
                pthread_mutex_unlock(&mutex);
                pthread_cond_broadcast(&cond_worker);
            }

        // END
        } else if (strcmp(cmd, "end") == 0 && arg1 && !arg2) {
            for (int i = 0; i < workers_count; i++){
                if (strcmp(workers[i].name, arg1) == 0){
                    pthread_mutex_lock(&mutex);
                    workers[i].leave = true;
                    occupation_places[workers[i].place]--;
                    pthread_mutex_unlock(&mutex);
                    pthread_cond_broadcast(&cond_worker);
                    break;
               }
            }
        // ADD
        } else if (strcmp(cmd, "add") == 0 && arg1 && !arg2) {
            int place = find_string_in_array(place_str, _PLACE_COUNT, arg1);
            if (place >= 0){
                pthread_mutex_lock(&mutex);
                ready_places[place]++;
                pthread_mutex_unlock(&mutex);
                pthread_cond_broadcast(&cond_worker);
            }

        // REMOVE
        } else if (strcmp(cmd, "remove") == 0 && arg1 && !arg2) {
            int place = find_string_in_array(place_str, _PLACE_COUNT, arg1);
            pthread_mutex_lock(&mutex);
            ready_places[place]--;
            pthread_mutex_unlock(&mutex);
        } 
        free(line);
    }

    // check the situation when there is job, but there is no workers for this job
    pthread_mutex_lock(&mutex);
    for (int phase_ind = _PHASE_COUNT - 1; phase_ind >= 0; phase_ind--)
        for (int product_ind = 0; product_ind < _PRODUCT_COUNT; product_ind++)
            if (parts[product_ind][phase_ind] > 0 && occupation_places[parts_places[product_ind][phase_ind]] == 0)
                parts[product_ind][phase_ind] = 0;
    pthread_mutex_unlock(&mutex);
    

    // waiting for exit
    exit_control();
    while (!exit_flag){
        pthread_mutex_lock(&mutex);
        pthread_cond_wait(&cond_worker, &mutex);
        exit_control();
        pthread_mutex_unlock(&mutex);
    }

    pthread_cond_broadcast(&cond_worker);

    for (int i = 0; i < workers_count; i++)
        pthread_join(workers[i].thread, NULL);
    
    // deallocation
    for (int i = 0; i < workers_count; i++)
        free(workers[i].name);
    free(workers);
    
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond_worker);

    return 0;
}


