#include "mzapo.h"

uint32_t left_led_colour_value = 0x00FFFF00;
uint32_t right_led_colour_value = 0x00FFFF00;

int font_size = 1;
unsigned char *mem_base;
uint32_t rgb_knobs_value;
int rk, gk, bk, rb, gb, bb;
int old_rk, old_gk, old_bk;
int check_rb = 0, check_gb = 0, check_bb = 0;

void parlcd_init()
{
	parlcd_mem_base = map_phys_address(PARLCD_REG_BASE_PHYS, PARLCD_REG_SIZE, 0);

	if (parlcd_mem_base == NULL)
	{
		printf("##Err: map_phys_address\n");
		exit(1);
	}
	parlcd_hx8357_init(parlcd_mem_base); // just one time after power-up
}

void led_init()
{
	mem_base = map_phys_address(SPILED_REG_BASE_PHYS, SPILED_REG_SIZE, 0);
	*(volatile uint32_t *)(mem_base + SPILED_REG_LED_LINE_o) = 0xFFFFFFFF; // light on all LEDs
	*(volatile uint32_t *)(mem_base + SPILED_REG_LED_RGB1_o) = 0xFFFFFFFF;
	*(volatile uint32_t *)(mem_base + SPILED_REG_LED_RGB2_o) = 0xFFFFFFFF;
}

void knob_values_update()
{
	rgb_knobs_value = *(volatile uint32_t *)(mem_base + SPILED_REG_KNOBS_8BIT_o);
	bk = rgb_knobs_value & 0xFF;					 // blue knob position
	gk = (rgb_knobs_value >> 8) & 0xFF;		 // green knob position
	rk = (rgb_knobs_value >> 16) & 0xFF;	 // red knob position
	int bbx = (rgb_knobs_value >> 24) & 1; // blue button
	int gbx = (rgb_knobs_value >> 25) & 1; // green button
	int rbx = (rgb_knobs_value >> 26) & 1; // red buttom
	rb = (!rbx && check_rb);
	gb = (!gbx && check_gb);
	bb = (!bbx && check_bb);

	check_rb = rbx;
	check_gb = gbx;
	check_bb = bbx;
}

void old_knob_values_update()
{
	old_bk = bk; // blue knob position
	old_gk = gk; // green knob position
	old_rk = rk; // red knob position
}

void background(uint16_t color)
{
	for (int i = 0; i < 320; ++i)
		for (int j = 0; j < 480; j++)
			frame[i][j] = color;
}

_Bool is_moved_left(int k, int old_k)
{
	if (k - old_k < -2)
		return 1;
	return 0;
}

_Bool is_moved_right(int k, int old_k)
{
	if (k - old_k > 2)
		return 1;
	return 0;
}

void led_update(struct HSV color0, struct HSV color1)
{
	struct RGB rgb = HSVToRGB(color0);
	*(volatile uint32_t *)(mem_base + SPILED_REG_LED_RGB1_o) = (rgb.B | rgb.G << 8 | rgb.R << 16);
	rgb = HSVToRGB(color1);
	*(volatile uint32_t *)(mem_base + SPILED_REG_LED_RGB2_o) = (rgb.B | rgb.G << 8 | rgb.R << 16);
}
