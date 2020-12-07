#include<stdio.h>
#include<stdlib.h> 
#include<unistd.h> 
#include<sys/types.h> 
#include<sys/wait.h> 
#include <signal.h>
#include <sys/stat.h> 
#include <fcntl.h>


int main(){
    int f = fork();
    if (f == 0){
        close(0);
        printf("Start\n");
        open("a.txt", O_WRONLY | O_CREAT);
        execl("./nsd", "./nsd", NULL);
    } else {
        wait(NULL);
    }
    return 0;
}
