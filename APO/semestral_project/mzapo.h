/*******************************************************************

APO semestral project 2019

Contributors:
- Karina Balagazova
- Lukas Frana

 *******************************************************************/

#ifndef MZAPO_H
#define MZAPO_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "lcdframe.h"
#include "mzapo_parlcd.h"
#include "mzapo_phys.h"
#include "mzapo_regs.h"
#include "hsv_to_rgb.h"

#ifdef __cplusplus
extern "C" {
#endif

extern uint32_t rgb_knobs_value;
extern uint32_t left_led_colour_value;
extern uint32_t right_led_colour_value;

extern unsigned char *mem_base;
extern int rk, gk, bk, rb, gb, bb;
extern int old_rk, old_gk, old_bk;
extern int select_menu_value;
extern int font_size;

void parlcd_init();
void led_init();
void led_update(struct HSV color0, struct HSV color1);
void knob_values_update();
void old_knob_values_update();
void background(uint16_t color);
_Bool is_moved_left(int k, int old_k);
_Bool is_moved_right(int k, int old_k);

#ifdef __cplusplus
} /* extern "C"*/
#endif

#endif /*MZAPO_H*/
