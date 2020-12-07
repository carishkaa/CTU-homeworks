#include <stdio.h>

extern "C" void *my_malloc(unsigned int size);
extern "C" int my_free(void *address);

void *brk(void *address){
    unsigned w0 = 3;
    unsigned w1 = (unsigned)address;
    asm volatile (
                  "   mov %%esp, %%ecx    ;"
                  "   mov $1f, %%edx      ;"
                  "   sysenter            ;"
                  "1:                     ;"
                  : "+a" (w0) : "S" (w1) : "ecx", "edx", "memory");
    return (void*)w0;
}

typedef struct Header{
    unsigned int size;
    unsigned int is_free;
    struct Header *next;
} header_t;

header_t *head = NULL, *tail = NULL;
void *break_current = NULL;

header_t *find_free_block(unsigned int size){
    unsigned int new_block_size = 0;
    header_t *new_block = NULL;
    
    header_t *curr = head;
    while(curr){
        if (!curr->is_free){
            new_block_size = 0;
            new_block = NULL;
        } else {
            // suitable block size
            if (curr->size >= size)
                return curr;
            
            // join empty blocks
            if (!new_block){
                new_block = curr;
                new_block_size = curr->size;
            } else {
                new_block_size += curr->size + sizeof(header_t);
            }
            
            if (new_block_size >= size){
                new_block->is_free = 1;
                new_block->size = new_block_size;
                new_block->next = curr->next;
                return new_block;
            }
        }
        curr = curr->next;
    }
    return NULL;
}

void *my_malloc(unsigned int size){
    if (!size)
        return NULL;
    
    // looking for a free block
    header_t *header = find_free_block(size);
    if (header){
        header->is_free = 0;
        return (void*)(header + 1);
    }
    
    // create a new block
    break_current = brk(0);
    void *block = brk((void*)((unsigned int)break_current + sizeof(header_t) + size));
    if (!block)
        return NULL;
    
    header = (header_t*)block;
    header->size = size;
    header->is_free = 0;
    header->next = NULL;
    
    if (!head)
        head = header;
    if (tail)
        tail->next = header;
    tail = header;
    return (void*)(header + 1);
}

int my_free(void *address){
    break_current = brk(0);
    if (address <= break_current){
        header_t *header = (header_t*)address - 1;
        if (!header->is_free){
            header->is_free = 1;
            return 0;
        }
    }
    return 1;
}

