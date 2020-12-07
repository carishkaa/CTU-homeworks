/**
 * Copied from mzapo_template
 */

#ifndef _LCDFRAME_H_
#define _LCDFRAME_H_

#include <stdint.h>

#define FRAME_H 320
#define FRAME_W 480
#define MARGIN 10

#ifdef __cplusplus
extern "C" {
#endif

extern unsigned char *parlcd_mem_base;

extern uint16_t frame [FRAME_H][FRAME_W];

void frame2lcd();
int char2frame(char c, int yrow, int xcolumn, uint16_t forecolor, uint16_t backcolor, int font_size);
int str2frame(char * str, int yrow, int xcolumn, uint16_t forecolor, uint16_t backcolor, int font_size); 
int str_width(char * str, int font_size);
void text_render(char *text, int x, int y, _Bool reversed);

#ifdef __cplusplus
} /* extern "C"*/
#endif

#endif
