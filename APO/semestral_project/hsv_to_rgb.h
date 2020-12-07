/*******************************************************************

APO semestral project 2019

Inspired by StackOverflow question

Contributors:
- Karina Balagazova
- Lukas Frana

 *******************************************************************/

#ifndef HSV_TO_RGB_H
#define HSV_TO_RGB_H

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

#ifdef __cplusplus
extern "C" {
#endif

struct RGB {
  unsigned char R;
  unsigned char G;
  unsigned char B;
};

struct HSV {
  short H;
  unsigned char S;
  unsigned char V;
};

struct RGB HSVToRGB(struct HSV hsv);

#ifdef __cplusplus
} /* extern "C"*/
#endif

#endif /*HSV_TO_RGB_H*/