#define _POSIX_C_SOURCE 200809L
#define MARGIN 10
#define DIFF 0.05
#include "menu.h"
#include "mzapo.h"
#include <time.h>
#include <math.h>

long start_time;
struct timespec spec;

void menu_render()
{
  for (int i = 0; i < menu->c_count; i++)
  {
    // Print menu item
    text_render(menu->c[i].label, 0, 40 + SIZE_Y * (i + 1), i == menu->selected_item);
  }
}

void ui_render()
{
  // char bar[80] = " Back | Okey | --- ";
  struct RGB rgb;
  uint16_t rect_color;

  // sprintf(bar, "Mode: %d", settings.mode);
  // text_render(bar, 0, 0, 1);

  rgb = HSVToRGB(settings.led0.color_min);
  rect_color = (uint16_t)(((rgb.R << 8) & 0xf800) + ((rgb.G << 3) & 0x07e0) + ((rgb.B >> 3) & 0x1f));
  rectangle_fill(0, 0, 40, 40, rect_color);

  rgb = HSVToRGB(settings.led0.color_max);
  rect_color = (uint16_t)(((rgb.R << 8) & 0xf800) + ((rgb.G << 3) & 0x07e0) + ((rgb.B >> 3) & 0x1f));
  rectangle_fill(40, 0, 40, 40, rect_color);

  rgb = HSVToRGB(settings.led1.color_min);
  rect_color = (uint16_t)(((rgb.R << 8) & 0xf800) + ((rgb.G << 3) & 0x07e0) + ((rgb.B >> 3) & 0x1f));
  rectangle_fill(400, 0, 40, 40, rect_color);

  rgb = HSVToRGB(settings.led1.color_max);
  rect_color = (uint16_t)(((rgb.R << 8) & 0xf800) + ((rgb.G << 3) & 0x07e0) + ((rgb.B >> 3) & 0x1f));
  rectangle_fill(440, 0, 40, 40, rect_color);
  
  // sprintf(bar, "Mode: %d", settings.mode);
  // text_render(bar, 0, 0, 1);
  // Render top bar
  // char bar[80];
  
  // text_render(bar, 0, SIZE_Y, 0);

  // Render bottom bar
  // text_render(bar, 0, 440, 1);

  char mode[20];
  sprintf(mode, "Mode: %d", settings.mode);
  text_render(mode, 0, 0, 1);
  // str2frame(mode, 0, 220, settings.color, settings.background, 2);

  str2frame("Back", 280, 85, settings.color, settings.background, 2);
  str2frame("Okey", 280, 240, settings.color, settings.background, 2);
  str2frame("----", 280, 420, settings.color, settings.background, 2);

  // Render menu or callback
  menu->c_count > 0 ? menu_render() : (*((*(menu)).callback))();
}

void start()
{
  printf("Hello MZAPO\n");

  // Default settings
  settings.font_size = 1;
  settings.color = 0x7FF;
  settings.background = 0x0;
  settings.mode = 0;

  settings.led0.color_min.H = 100;
  settings.led0.color_min.S = 100;
  settings.led0.color_min.V = 100;

  settings.led0.color_max.H = 150;
  settings.led0.color_max.S = 100;
  settings.led0.color_max.V = 100;

  settings.led0.period = 0.0;
  settings.led0.millis = 1000;
  settings.led0.on_time = 1000;
  settings.led0.off_time = 100;
  settings.led0.phase = 0;

  settings.led1.color_min.H = 360;
  settings.led1.color_min.S = 100;
  settings.led1.color_min.V = 100;

  settings.led1.color_max.H = 0;
  settings.led1.color_max.S = 100;
  settings.led1.color_max.V = 100;

  settings.led1.period = 0.0;
  settings.led1.millis = 500;
  settings.led1.on_time = 100;
  settings.led1.off_time = 100;
  settings.led1.phase = 0;

  menu_init();
  parlcd_init();
  led_init();

  // Select menu root as current menu
  menu = &menu_root;
  clock_gettime(CLOCK_MONOTONIC, &spec);

  start_time = (long)(spec.tv_sec * 1000 + floor(spec.tv_nsec / 1.0e6));
}

void loop()
{
  knob_values_update();

  clock_gettime(CLOCK_MONOTONIC, &spec);
  long curr_time = spec.tv_sec * 1000 + floor(spec.tv_nsec / 1.0e6);
  long diff_time, millis, on_time, off_time;
  unsigned char led0_is_on = 1, led1_is_on = 1;

  // Calc LED0 period
  millis = settings.led0.millis;
  diff_time = ((int)(curr_time - start_time)) % (millis * 2);
  if (diff_time > millis)
  {
    settings.led0.period = (2 * millis - diff_time) / (double)millis;
    // printf("+%f\n", settings.led0.period);
  }
  else
  {
    settings.led0.period = diff_time / (double)millis;
    // printf("-%f\n", settings.led0.period);
  }

  // Calc LED0 on/off
  on_time = settings.led0.on_time;
  off_time = settings.led0.off_time;
  diff_time = ((int)(curr_time - start_time + settings.led0.phase)) % (on_time + off_time);
  led0_is_on = (on_time >= diff_time);

  // Calc LED1 period
  millis = settings.led1.millis;
  diff_time = ((int)(curr_time - start_time)) % (millis * 2);
  if (diff_time > millis)
  {
    settings.led1.period = (2 * millis - diff_time) / (double)millis;
    // printf("+%f\n", settings.led1.period);
  }
  else
  {
    settings.led1.period = diff_time / (double)millis;
    // printf("-%f\n", settings.led1.period);
  }

  // Calc LED1 on/off
  on_time = settings.led1.on_time;
  off_time = settings.led1.off_time;
  diff_time = ((int)(curr_time - start_time + settings.led1.phase)) % (on_time + off_time);
  led1_is_on = (on_time >= diff_time);

  struct HSV color0;
  struct HSV color1;
  struct HSV color_off;

  color_off.H = color_off.S = color_off.V = 0;

  color0.H = (settings.led0.color_max.H - settings.led0.color_min.H) * settings.led0.period + settings.led0.color_min.H;
  color0.S = (settings.led0.color_max.S - settings.led0.color_min.S) * settings.led0.period + settings.led0.color_min.S;
  color0.V = (settings.led0.color_max.V - settings.led0.color_min.V) * settings.led0.period + settings.led0.color_min.V;

  if (settings.mode == 1) {
    settings.led1.period = settings.led0.period;
    settings.led1.color_min = settings.led0.color_min;
    settings.led1.color_max = settings.led0.color_max;
  } else if (settings.mode == 2) {
    settings.led1.period = 1 - settings.led0.period;
    settings.led1.color_min = settings.led0.color_min;
    settings.led1.color_max = settings.led0.color_max;
  }

  color1.H = (settings.led1.color_max.H - settings.led1.color_min.H) * settings.led1.period + settings.led1.color_min.H;
  color1.S = (settings.led1.color_max.S - settings.led1.color_min.S) * settings.led1.period + settings.led1.color_min.S;
  color1.V = (settings.led1.color_max.V - settings.led1.color_min.V) * settings.led1.period + settings.led1.color_min.V;

  led_update(led0_is_on ? color0 : color_off, led1_is_on ? color1 : color_off);

  parlcd_clear();
  menu_update();
  ui_render();
  old_knob_values_update();
  frame2lcd();
}

int main(int argc, char *argv[])
{
  start();

  while (1)
    loop();

  return 0;
}