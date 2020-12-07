#include "hsv_to_rgb.h"

struct RGB HSVToRGB(struct HSV hsv) {
  struct RGB rgb;

  if (hsv.S == 0) {
    rgb.R = hsv.V;
    rgb.G = hsv.V;
    rgb.B = hsv.V;
  } else {
    unsigned char i;
    double f, p, q, t, h, s, v;

    h = hsv.H;
    s = hsv.S / 100.0;
    v = hsv.V / 100.0;

    i = ((int)trunc(h / 60));

    if (h == 360)
      f = 0;
    else
      f = h / 60 - i;

    p = v * (1 - s);
    q = v * (1 - (s * f));
    t = v * (1 - (s * (1 - f)));

    switch (i) {
    case 0:
      rgb.R = v * 255;
      rgb.G = t * 255;
      rgb.B = p * 255;
      break;

    case 1:
      rgb.R = q * 255;
      rgb.G = v * 255;
      rgb.B = p * 255;
      break;

    case 2:
      rgb.R = p * 255;
      rgb.G = v * 255;
      rgb.B = t * 255;
      break;

    case 3:
      rgb.R = p * 255;
      rgb.G = q * 255;
      rgb.B = v * 255;
      break;

    case 4:
      rgb.R = t * 255;
      rgb.G = p * 255;
      rgb.B = v * 255;
      break;

    default:
      rgb.R = v * 255;
      rgb.G = p * 255;
      rgb.B = q * 255;
      break;
    }
  }
  return rgb;
}