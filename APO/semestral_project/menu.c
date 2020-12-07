#define MILLIS 250
#define ON_OFF_TIME 100
#include "menu.h"
#include "hsv_to_rgb.h"
#include "mzapo.h"
#include <string.h>

Menu menu_root;
Menu *menu;
Settings settings;

unsigned char selected = 0;
unsigned char on_off_phase_time = 1;
_Bool led_min = 0;
_Bool led_max = 0;

void menu_millis_selector() {
  char *text = (char *)calloc(100, 1);
  strcat(text, "MODE:  ");
  strcat(text, menu->parent->label);
  strcat(text, " - ");
  strcat(text, menu->label);

  text_render(text, 0, 40, 0);

  if (selected == 0 || selected == 2) {
    if (is_moved_left(bk, old_bk) && settings.led0.millis > MILLIS)
      settings.led0.millis -= MILLIS;

    if (is_moved_right(bk, old_bk))
      settings.led0.millis += MILLIS;
  }

  if (selected == 1 || selected == 2) {
    if (is_moved_left(bk, old_bk) && settings.led1.millis > MILLIS) {
      // printf("%d ... <\n", is_moved_left(bk, old_bk));
      settings.led1.millis -= MILLIS;
    }

    if (is_moved_right(bk, old_bk)) {
    //   printf("%d ... >\n", is_moved_left(bk, old_bk));
      settings.led1.millis += MILLIS;
    }
  }

  char hsv_format_text[50] = "MILLIS: %d";
  char *hsv_text = (char *)calloc(strlen(hsv_format_text) + 10, 1);

  if (selected)
    sprintf(hsv_text, hsv_format_text, settings.led1.millis);
  else
    sprintf(hsv_text, hsv_format_text, settings.led0.millis);

  str2frame(hsv_text, 180, (480 - str_width(hsv_format_text, 2)) / 2 - 1, 0x0,
            0x7FF, 2);
}

void menu_time_selector() {
  char *text = (char *)calloc(100, 1);
  strcat(text, "MODE:  ");
  strcat(text, menu->parent->label);
  strcat(text, " - ");
  strcat(text, menu->label);

  text_render(text, 0, 40, 0);

  if (selected == 0 || selected == 2) {
    if (is_moved_left(bk, old_bk)) {
      if (on_off_phase_time == 0 && settings.led0.on_time > ON_OFF_TIME)
        settings.led0.on_time -= ON_OFF_TIME;
      if (on_off_phase_time == 1 && settings.led0.off_time >= ON_OFF_TIME)
        settings.led0.off_time -= ON_OFF_TIME;
      if (on_off_phase_time == 2 && settings.led0.phase >= ON_OFF_TIME)
        settings.led0.phase -= ON_OFF_TIME;
      if (on_off_phase_time == 3 && settings.led0.millis > ON_OFF_TIME)
        settings.led0.millis -= ON_OFF_TIME;
    }

    if (is_moved_right(bk, old_bk)) {
      if (on_off_phase_time == 0)
        settings.led0.on_time += ON_OFF_TIME;
      if (on_off_phase_time == 1)
        settings.led0.off_time += ON_OFF_TIME;
      if (on_off_phase_time == 2)
        settings.led0.phase += ON_OFF_TIME;
      if (on_off_phase_time == 3)
        settings.led0.phase += ON_OFF_TIME;
    }
  }

  if (selected == 1 || selected == 2) {
    if (is_moved_left(bk, old_bk)) {
      if (on_off_phase_time == 0 && settings.led1.on_time > ON_OFF_TIME)
        settings.led1.on_time -= ON_OFF_TIME;
      if (on_off_phase_time == 1 && settings.led1.off_time >= ON_OFF_TIME)
        settings.led1.off_time -= ON_OFF_TIME;
      if (on_off_phase_time == 2 && settings.led1.phase >= ON_OFF_TIME)
        settings.led1.phase -= ON_OFF_TIME;
      if (on_off_phase_time == 3 && settings.led1.millis > ON_OFF_TIME)
        settings.led1.millis -= ON_OFF_TIME;
    }

    if (is_moved_right(bk, old_bk)) {
      if (on_off_phase_time == 0)
        settings.led1.on_time += ON_OFF_TIME;
      if (on_off_phase_time == 1)
        settings.led1.off_time += ON_OFF_TIME;
      if (on_off_phase_time == 2)
        settings.led1.phase += ON_OFF_TIME;
      if (on_off_phase_time == 3)
        settings.led1.millis += ON_OFF_TIME;
    }
  }

  char hsv_format_text[50] = "MILLIS: %d";
  char *hsv_text = (char *)calloc(strlen(hsv_format_text) + 10, 1);

  if (selected) {
    if (on_off_phase_time == 0)
        sprintf(hsv_text, hsv_format_text, settings.led1.on_time);
      if (on_off_phase_time == 1)
        sprintf(hsv_text, hsv_format_text, settings.led1.off_time);
      if (on_off_phase_time == 2)
        sprintf(hsv_text, hsv_format_text, settings.led1.phase);
      if (on_off_phase_time == 3)
        sprintf(hsv_text, hsv_format_text, settings.led1.millis);
  } else {
    if (on_off_phase_time == 0)
        sprintf(hsv_text, hsv_format_text, settings.led0.on_time);
      if (on_off_phase_time == 1)
        sprintf(hsv_text, hsv_format_text, settings.led0.off_time);
      if (on_off_phase_time == 2)
        sprintf(hsv_text, hsv_format_text, settings.led0.phase);
      if (on_off_phase_time == 3)
        sprintf(hsv_text, hsv_format_text, settings.led0.millis);
  }

  str2frame(hsv_text, 180, (480 - str_width(hsv_format_text, 2)) / 2 - 1, 0x0, 0x7FF, 2);
}

void set_led0_min1_max1() {
  selected = 0;
  led_min = led_max = 1;

  menu_color_selector();
}

void set_led0_min1_max0() {
  selected = 0;
  led_min = 1;
  led_max = 0;

  menu_color_selector();
}

void set_led0_min0_max1() {
  selected = 0;
  led_min = 0;
  led_max = 1;

  menu_color_selector();
}

void set_led1_min1_max1() {
  selected = 1;
  led_min = led_max = 1;

  menu_color_selector();
}

void set_led1_min1_max0() {
  selected = 1;
  led_min = 1;
  led_max = 0;

  menu_color_selector();
}

void set_led1_min0_max1() {
  selected = 1;
  led_min = 0;
  led_max = 1;

  menu_color_selector();
}

void set_led2_min1_max1() {
  selected = 2;
  led_min = led_max = 1;

  menu_color_selector();
}

void set_led2_min1_max0() {
  selected = 2;
  led_min = 1;
  led_max = 0;

  menu_color_selector();
}

void set_led2_min0_max1() {
  selected = 2;
  led_min = 0;
  led_max = 1;

  menu_color_selector();
}

void set_led0_millis() {
  selected = 0;
  on_off_phase_time = 3;

  menu_time_selector();
}

void set_led1_millis() {
  selected = 1;
  on_off_phase_time = 3;

  menu_time_selector();
}

void set_led2_millis() {
  selected = 2;
  on_off_phase_time = 3;

  menu_time_selector();
}

void set_led0_on_time() {
  selected = 0;
  on_off_phase_time = 0;

  menu_time_selector();
}

void set_led0_off_time() {
  selected = 0;
  on_off_phase_time = 1;

  menu_time_selector();
}

void set_led0_phase_time() {
  selected = 0;
  on_off_phase_time = 2;

  menu_time_selector();
}

void set_led1_on_time() {
  selected = 1;
  on_off_phase_time = 0;

  menu_time_selector();
}

void set_led1_off_time() {
  selected = 1;
  on_off_phase_time = 1;

  menu_time_selector();
}

void set_led1_phase_time() {
  selected = 1;
  on_off_phase_time = 2;

  menu_time_selector();
}

void set_led2_on_time() {
  selected = 2;
  on_off_phase_time = 0;

  menu_time_selector();
}

void set_led2_off_time() {
  selected = 2;
  on_off_phase_time = 1;

  menu_time_selector();
}

void set_led2_phase_time() {
  selected = 2;
  on_off_phase_time = 2;

  menu_time_selector();
}

void set_normal_mode() {
  settings.mode = 0;

  menu = menu->parent;
}

void set_copy_mode() {
  settings.mode = 1;

  menu = menu->parent;
}

void set_anticopy_mode() {
  settings.mode = 2;

  menu = menu->parent;
}

void menu_init() {
  // Root
  menu_set(&menu_root, "Root", 2, NULL, NULL);

  // Root - Settings
  menu_set(&(menu_root.c[0]), "Settings", 3, &menu_root, NULL);

  // Root - Settings - Font size
  menu_set(&(menu_root.c[0].c[0]), "Font size", 0, &(menu_root.c[0]), menu_0_0_font_size);

  // Root - Settings - Mode
  menu_set(&(menu_root.c[0].c[2]), "Mode", 3, &(menu_root.c[0]), NULL);

  // Root - Settings - Mode - Normal
  menu_set(&(menu_root.c[0].c[2].c[0]), "Normal", 0, &(menu_root.c[0].c[2]), set_normal_mode);

  // Root - Settings - Mode - Copy
  menu_set(&(menu_root.c[0].c[2].c[1]), "Copy", 0, &(menu_root.c[0].c[2]), set_copy_mode);

  // Root - Settings - Mode - Anticopy
  menu_set(&(menu_root.c[0].c[2].c[2]), "Anticopy", 0, &(menu_root.c[0].c[2]), set_anticopy_mode);

  // Root - Settings - LED settings
  menu_set(&(menu_root.c[0].c[1]), "LED settings", 3, &(menu_root.c[0]), NULL);

  // Root - Settings - LED settings - LED1
  menu_set(&(menu_root.c[0].c[1].c[0]), "LED 1", 3, &(menu_root.c[0].c[1]), NULL);

  // Root - Settings - LED settings - LED1 - STATIC
  menu_set(&(menu_root.c[0].c[1].c[0].c[0]), "STATIC", 0, &(menu_root.c[0].c[1].c[0]), set_led0_min1_max1);

  // Root - Settings - LED settings - LED1 - DYNAMIC
  menu_set(&(menu_root.c[0].c[1].c[0].c[1]), "DYNAMIC", 3, &(menu_root.c[0].c[1].c[0]), NULL);

  // Root - Settings - LED settings - LED1 - DYNAMIC - MIN COLOR
  menu_set( &(menu_root.c[0].c[1].c[0].c[1].c[0]), "MIN COLOR", 0, &(menu_root.c[0].c[1].c[0].c[1]), set_led0_min1_max0);

  // Root - Settings - LED settings - LED1 - DYNAMIC - MAX COLOR
  menu_set( &(menu_root.c[0].c[1].c[0].c[1].c[1]), "MAX COLOR", 0, &(menu_root.c[0].c[1].c[0].c[1]), set_led0_min0_max1);

  // Root - Settings - LED settings - LED1 - DYNAMIC - PERIOD
  menu_set( &(menu_root.c[0].c[1].c[0].c[1].c[2]), "PERIOD", 0, &(menu_root.c[0].c[1].c[0].c[1]), set_led0_millis);

  // Root - Settings - LED settings - LED1 - BLINKING
  menu_set(&(menu_root.c[0].c[1].c[0].c[2]), "BLINK", 3, &(menu_root.c[0].c[1].c[0]), NULL);

  // Root - Settings - LED settings - LED1 - BLINKING - ON TIME
  menu_set( &(menu_root.c[0].c[1].c[0].c[2].c[0]), "ON TIME", 0, &(menu_root.c[0].c[1].c[0].c[2]), set_led0_on_time);

  // Root - Settings - LED settings - LED1 - BLINKING - OFF TIME
  menu_set( &(menu_root.c[0].c[1].c[0].c[2].c[1]), "OFF TIME", 0, &(menu_root.c[0].c[1].c[0].c[2]), set_led0_off_time);

  // Root - Settings - LED settings - LED1 - BLINKING - PHASE
  menu_set( &(menu_root.c[0].c[1].c[0].c[2].c[2]), "PHASE", 0, &(menu_root.c[0].c[1].c[0].c[2]), set_led0_phase_time);

  // Root - Settings - LED settings - LED2
  menu_set(&(menu_root.c[0].c[1].c[1]), "LED 2", 3, &(menu_root.c[0].c[1]), NULL);

  // Root - Settings - LED settings - LED2 - STATIC
  menu_set(&(menu_root.c[0].c[1].c[1].c[0]),"STATIC", 0, &(menu_root.c[0].c[1].c[1]), NULL);

  // Root - Settings - LED settings - LED2 - DYNAMIC
  menu_set(&(menu_root.c[0].c[1].c[1].c[1]),"DYNAMIC", 3, &(menu_root.c[0].c[1].c[1]),NULL);

  // Root - Settings - LED settings - LED2 - DYNAMIC - MIN COLOR
  menu_set( &(menu_root.c[0].c[1].c[1].c[1].c[0]), "MIN COLOR", 0, &(menu_root.c[0].c[1].c[1].c[1]), set_led1_min1_max0);

  // Root - Settings - LED settings - LED2 - DYNAMIC - MAX COLOR
  menu_set(&(menu_root.c[0].c[1].c[1].c[1].c[1]),"MAX COLOR", 0,&(menu_root.c[0].c[1].c[1].c[1]),set_led1_min0_max1);

  // Root - Settings - LED settings - LED2 - DYNAMIC - PERIOD
  menu_set(&(menu_root.c[0].c[1].c[1].c[1].c[2]),"PERIOD", 0, &(menu_root.c[0].c[1].c[1].c[1]),set_led1_millis);

  // Root - Settings - LED settings - LED2 - BLINKING
  menu_set(&(menu_root.c[0].c[1].c[1].c[2]), "BLINK", 3, &(menu_root.c[0].c[1].c[1]), NULL);

  // Root - Settings - LED settings - LED2 - BLINKING - ON TIME
  menu_set( &(menu_root.c[0].c[1].c[1].c[2].c[0]), "ON TIME", 0, &(menu_root.c[0].c[1].c[1].c[2]), set_led1_on_time);

  // Root - Settings - LED settings - LED2 - BLINKING - OFF TIME
  menu_set( &(menu_root.c[0].c[1].c[1].c[2].c[1]), "OFF TIME", 0, &(menu_root.c[0].c[1].c[1].c[2]), set_led1_off_time);

  // Root - Settings - LED settings - LED2 - BLINKING - PHASE
  menu_set( &(menu_root.c[0].c[1].c[1].c[2].c[2]), "PHASE", 0, &(menu_root.c[0].c[1].c[1].c[2]), set_led1_phase_time);

  // Root - Settings - LED settings - BOTH
  menu_set(&(menu_root.c[0].c[1].c[2]), "BOTH", 3,  &(menu_root.c[0].c[1]), NULL);

  // Root - Settings - LED settings - BOTH - STATIC
  menu_set(&(menu_root.c[0].c[1].c[2].c[0]), "STATIC", 0, &(menu_root.c[0].c[1].c[2]), set_led2_min1_max1);

  // Root - Settings - LED settings - BOTH - DYNAMIC
  menu_set(&(menu_root.c[0].c[1].c[2].c[1]), "DYNAMIC", 3, &(menu_root.c[0].c[1].c[2]), NULL);

  // Root - Settings - LED settings - BOTH - DYNAMIC - MIN COLOR
  menu_set(&(menu_root.c[0].c[1].c[2].c[1].c[0]),"MIN COLOR", 0,&(menu_root.c[0].c[1].c[2].c[1]),set_led2_min1_max0);

  // Root - Settings - LED settings - BOTH - DYNAMIC - MAX COLOR
  menu_set(&(menu_root.c[0].c[1].c[2].c[1].c[1]),"MAX COLOR", 0,&(menu_root.c[0].c[1].c[2].c[1]),set_led2_min0_max1);

  // Root - Settings - LED settings - BOTH - DYNAMIC - PERIOD
  menu_set(&(menu_root.c[0].c[1].c[2].c[1].c[2]),"PERIOD", 0, &(menu_root.c[0].c[1].c[2].c[1]),set_led2_millis);

  // Root - Settings - LED settings - BOTH - BLINKING
  menu_set(&(menu_root.c[0].c[1].c[2].c[2]), "BLINK", 3, &(menu_root.c[0].c[1].c[2]), NULL);

  // Root - Settings - LED settings - BOTH - BLINKING - ON TIME
  menu_set( &(menu_root.c[0].c[1].c[2].c[2].c[0]), "ON TIME", 0, &(menu_root.c[0].c[1].c[2].c[2]), set_led2_on_time);

  // Root - Settings - LED settings - BOTH - BLINKING - OFF TIME
  menu_set( &(menu_root.c[0].c[1].c[2].c[2].c[1]), "OFF TIME", 0, &(menu_root.c[0].c[1].c[2].c[2]), set_led2_off_time);

  // Root - Settings - LED settings - BOTH - BLINKING - PHASE
  menu_set( &(menu_root.c[0].c[1].c[2].c[2].c[2]), "PHASE", 0, &(menu_root.c[0].c[1].c[2].c[2]), set_led2_phase_time);

  // Root - Exit
  menu_set(&(menu_root.c[1]), "Exit", 0, &menu_root, menu_1_exit);
}

void menu_set(Menu *menu, char *label, int c_count, Menu *parent,
              void (*callback)()) {
  menu->label = label;
  menu->c_count = c_count;
  menu->selected_item = 0;
  menu->parent = parent;
  menu->callback = callback;
  menu->c = malloc(sizeof(Menu) * c_count);
}

void menu_update() {
  int si = menu->selected_item;

  if (is_moved_left(gk, old_gk))
    si--;

  if (is_moved_right(gk, old_gk))
    si++;

  menu->selected_item = si < 0 ? (menu->c_count - 1)
                               : (si > (menu->c_count - 1) ? 0 : si);

  // Move to parent menu
  if (rb == 1 && menu->parent != NULL) {
    menu = menu->parent;
  }

  // Move to child menu
  if (gb == 1 && menu->c_count > 0) {
    menu = &(menu->c[menu->selected_item]);
  }
}

void menu_color_selector() {
  struct RGB rgb;
  struct HSV hsv;

  char *text = (char *)calloc(100, 1);
  strcat(text, menu->parent->label);
  strcat(text, " - ");
  strcat(text, menu->label);

  text_render(text, 0, 40, 0);

  hsv.H = (rk * 360) / 255;
  hsv.S = (gk * 100) / 255;
  hsv.V = (bk * 100) / 255;

  rgb = HSVToRGB(hsv);
  uint16_t rect_color =
      (uint16_t)(((rgb.R << 8) & 0xf800) + ((rgb.G << 3) & 0x07e0) +
                 ((rgb.B >> 3) & 0x1f));

  // if (is_moved_left(bk, old_bk) || is_moved_right(bk, old_bk) ||
  // is_moved_left(gk, old_gk) || is_moved_right(gk, old_gk) ||
  // is_moved_left(rk, old_rk) || is_moved_right(rk, old_rk))
  // {
  if (selected == 0 || selected == 2) {
    if (led_min)
      settings.led0.color_min = hsv;

    if (led_max)
      settings.led0.color_max = hsv;
  }

  if (selected == 1 || selected == 2) {
    if (led_min)
      settings.led1.color_min = hsv;

    if (led_max)
      settings.led1.color_max = hsv;
  }

  // printf("R: %d; G: %d; B: %d\n", rgb.R, rgb.G, rgb.B);
  // }

  rectangle_fill((int)((480 - 60) / 2) - 1, 90, 60, 60, rect_color);

  char hsv_format_text[50] = "H: %d S: %d V: %d";
  char *hsv_text = (char *)calloc(strlen(hsv_format_text) + 10, 1);
  sprintf(hsv_text, hsv_format_text, (int)hsv.H, (int)hsv.S, (int)hsv.V);
  str2frame(hsv_text, 180, (480 - str_width(hsv_format_text, 2)) / 2 - 1, 0x0,
            0x7FF, 2);
}

void rectangle_fill(int x0, int y0, int w, int h, uint16_t color) {
  for (int i = y0; i < y0 + h; i++)
    for (int j = x0; j < x0 + w; j++)
      frame[i][j] = color;
}

void menu_0_0_font_size() {
  if (settings.font_size == 1)
    text_render("Font size: normal", 0, SIZE_Y, 0);
  else
    text_render("Font size: double", 0, SIZE_Y, 0);

  if (is_moved_left(bk, old_bk) || is_moved_right(bk, old_bk))
    settings.font_size = settings.font_size == 1 ? 2 : 1;
}

void menu_1_exit() {
  parlcd_clear();
  str2frame("Goodbye!", 50, 50, 0x1F, 0xFFFF, 2);
  end();
  exit(0);
}

void end() {
  frame2lcd();
  printf("Goodbye MZAPO\n");
}
