/*******************************************************************

APO semestral project 2019

Contributors:
- Karina Balagazova
- Lukas Frana

 *******************************************************************/

#ifndef MENU_H
#define MENU_H

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

#define SIZE_Y 20*settings.font_size
#define parlcd_clear() background(0x0)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Menu {
  char *label;
  int y;
  struct Menu *c;
  int c_count;
  int selected_item;
  struct Menu *parent;
  void (*callback)();
} Menu;

typedef struct {
  struct HSV color_max;
  struct HSV color_min;
  double period;
  int millis;
  int on_time;
  int off_time;
  int phase;
} LED;

typedef struct {
  LED led0;
  LED led1;
  int mode;
  int selected;
  int font_size;
  uint16_t color;
  uint16_t background;
} Settings;

extern Menu menu_root;
extern Menu *menu;
extern Settings settings;

void menu_init();
void menu_set(Menu *menu, char *label, int c_count, Menu *parent, void (*callback)());
void menu_update();
void menu_color_selector();
void rectangle_fill(int x0, int y0, int w, int h, uint16_t color);
void menu_0_0_font_size();
void menu_1_exit();
void end();

#ifdef __cplusplus
} /* extern "C"*/
#endif

#endif /*MENU_H*/