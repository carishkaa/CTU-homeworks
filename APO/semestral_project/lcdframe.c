#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

#include "mzapo_parlcd.h"
#include "mzapo_phys.h"
#include "mzapo_regs.h"
#include "lcdframe.h"
#include "font_types.h"
#include "menu.h"

unsigned char *parlcd_mem_base=NULL;

uint16_t frame [FRAME_H][FRAME_W];

void frame2lcd()
{
	// lcd pointer to 0,0
	*(volatile uint16_t*)(parlcd_mem_base + PARLCD_REG_CMD_o) = 0x2c; // command
	
	volatile uint32_t* ptr = (volatile uint32_t*) frame;
	volatile uint32_t* dst = (volatile uint32_t*) (parlcd_mem_base + PARLCD_REG_DATA_o);
	
	int i;
	for(i=0; i<FRAME_H*(FRAME_W/2);i++) *dst = *ptr++;
}

int char2frame(char c, int yrow, int xcolumn, uint16_t forecolor, uint16_t backcolor, int font_size)
{
	int cix = c-' ';
	int w = 0;
	int y,x;
		w = font_size*(font_winFreeSystem14x16.width[cix]+4);
		for(y=0; y<16*font_size; y+=font_size){
			uint16_t mask = font_winFreeSystem14x16.bits[16*cix+y/font_size];
			for(x=0; x<w; x+=font_size){	
				for (int i = 0; i < font_size; i++)
					for (int j = 0; j < font_size; j++)
						frame[yrow+y+i][xcolumn+x+j]=(mask & 0x8000) ? forecolor : backcolor;
				mask = mask << 1;
			}
		}
	return w; 
}

int str2frame(char * str, int yrow, int xcolumn, uint16_t forecolor, uint16_t backcolor, int font_size) 
{
  char c; int w=0;
  while((c=*str++)!=0) 
     w += char2frame(c,yrow,xcolumn+w,forecolor,backcolor,font_size);
  return w;
}

int str_width(char * str, int font_size){
  char c; 
  int w=0;
  while((c=*str++)!=0){
  	int cix = c-' ';
    w += font_size*(font_winFreeSystem14x16.width[cix]+4);  
  }
  return w;
}

void text_render(char *text, int x, int y, _Bool reversed) {
  str2frame(text, MARGIN + y, (480 - str_width(text, settings.font_size))/2 - 1,
            reversed ? settings.background : settings.color,
            reversed ? settings.color : settings.background,
            settings.font_size);
}
