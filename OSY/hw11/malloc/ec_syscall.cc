#include "ec.h"
#include "ptab.h"
#include "bits.h"
#include <string.h>


typedef enum {
    sys_print      = 1,
    sys_sum        = 2,
    sys_break      = 3,
    sys_thr_create = 4,
    sys_thr_yield  = 5,
} Syscall_numbers;


void Ec::syscall_handler (uint8 a)
{
    // Get access to registers stored during entering the system - see
    // entry_sysenter in entry.S
    Sys_regs * r = current->sys_regs();
    Syscall_numbers number = static_cast<Syscall_numbers> (a);
    
    switch (number) {
        case sys_print: {
            char *data = reinterpret_cast<char*>(r->esi);
            unsigned len = r->edi;
            for (unsigned i = 0; i < len; i++)
                printf("%c", data[i]);
            break;
        }
        case sys_sum: {
            // Naprosto nepotřebné systémové volání na sečtení dvou čísel
            int first_number = r->esi;
            int second_number = r->edi;
            r->eax = first_number + second_number;
            break;
        }
            
        case sys_break: {
            mword address = r->esi;
            r->eax = Ec::break_current;
            if(!address) break;
            if (address < Ec::break_min || address > 0xC0000000){
                r->eax = 0;
                break;
            }
            mword cur_pages_break = align_up(Ec::break_current, PAGE_SIZE);
            mword new_pages_break = align_up(address, PAGE_SIZE);
            mword offset;
            
            // alloc pages
            if (address > Ec::break_current){
                mword orig_pages_break = cur_pages_break;
                // reset the unused part of page
                if ((offset = Ec::break_current % PAGE_SIZE) != 0){
                    memset(reinterpret_cast<void*>(Ec::break_current), 0, PAGE_SIZE - offset);
                }
                // allocation
                while(cur_pages_break != new_pages_break){
                    void *page = Kalloc::allocator.alloc_page(1, Kalloc::FILL_0);
                    if (page == NULL) break;
                    if (!Ptab::insert_mapping(cur_pages_break, Kalloc::virt2phys(page), Ptab::PRESENT | Ptab::RW | Ptab::USER)){
                        Kalloc::allocator.free_page(page);
                        break;
                    }
                    cur_pages_break += PAGE_SIZE;
                }
                // if memory fail -> free allocated pages
                if (cur_pages_break != new_pages_break){
                    new_pages_break = orig_pages_break;
                    r->eax = 0;
                }
            }
            
            // free pages
            if (address < Ec::break_current || r->eax == 0)
                while(cur_pages_break != new_pages_break){
                    cur_pages_break -= PAGE_SIZE;
                    void *p_free = Kalloc::phys2virt(Ptab::get_mapping(cur_pages_break) & ~PAGE_MASK);
                    Ptab::insert_mapping(cur_pages_break, Kalloc::virt2phys(p_free), 0);
                    Kalloc::allocator.free_page(p_free);
                }
            
            if (r->eax != 0) Ec::break_current = address;
            break;
        }
            
        default:
            printf ("unknown syscall %d\n", number);
            break;
    };
    
    ret_user_sysexit();
}
