#define SYS_EXIT 1
#define SYS_READ 3
#define SYS_WRITE 4

#define STDIN_FILENO 0
#define STDOUT_FILENO 1

#define NULL 0

static inline long syscall(long syscall, long arg1, long arg2, long arg3);

int write(int fd, const void *buf, int count);
int read(int fd, void *buf, int count);
void __exit(int status);


int isnum(char ch){
    return ch >= '0' && ch <= '9';
}

int isspc(char ch){
    return ch == ' ' || ch == '\n';
}

static void print(unsigned num){
    char buf_tmp[20];
    int count_tmp = 0;
    do {
        int alphabet = num%16;
        buf_tmp[count_tmp++] = (alphabet < 10) ? (alphabet + 48) : (alphabet + 87);
        num /= 16;
    } while (num);
    
    char buf[count_tmp + 2];
    int count = 0;
    buf[count++] = '0';
    buf[count++] = 'x';
    while(count_tmp > 0)
        buf[count++] = buf_tmp[--count_tmp];
    buf[count++] = '\n';
    
    int ret = write(STDOUT_FILENO, buf, count);
    if (ret == -1)
        __exit(1);
}

int _start()
{
    char buf[20];
    unsigned num = 0;
    int i;
    int num_digits = 0;
    unsigned chars_in_buffer = 0;
    
    for (/* no init */; /* no end condition */; i++, chars_in_buffer--) {
        if (chars_in_buffer == 0) {
            int ret = read(STDIN_FILENO, buf, sizeof(buf));
            if (ret < 0)
                __exit(1);
            i = 0;
            chars_in_buffer = ret;
        }
        if (num_digits > 0 && (chars_in_buffer == 0 /* EOF */ || !isnum(buf[i]))) {
            print(num);
            num_digits = 0;
            num = 0;
        }
        if (chars_in_buffer == 0 /* EOF */ || (!isspc(buf[i]) && !isnum(buf[i])))
            __exit(0); 
        
        if (isnum(buf[i])) {
            num = num * 10 + buf[i] - '0';
            num_digits++;
        }
    }
    __exit(0);
}


static inline long syscall(long syscall, long arg1, long arg2, long arg3) {
    long ret;
    asm volatile (
                  "int $0x80"
                  : "=a" (ret)
                  : "a" (syscall), "b"(arg1), "c"(arg2), "d" (arg3)
                  : "memory");
    return ret;
}

int write(int fd, const void *buf, int count){
    return syscall(SYS_WRITE, fd, (long)buf, count);
}

int read(int fd, void *buf, int count){
    return syscall(SYS_READ, fd, (long)buf, count);
}

void __exit(int status) {
    syscall(SYS_EXIT, status, NULL, NULL);
}
