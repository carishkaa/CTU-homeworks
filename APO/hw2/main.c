#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]){
    int MAX_COLOR_VALUE = 255;
    int width = 0, height = 0;
    
    //open input image
    FILE *input_file = fopen(argv[1], "rb");
    
    //read input image
    fscanf(input_file, "P6\n%d\n%d\n%d\n", &width, &height, &MAX_COLOR_VALUE);
    unsigned char *img = (unsigned char*)malloc(sizeof(unsigned char)* 3 * width * height);
    unsigned char *new_img = (unsigned char*)malloc(sizeof(unsigned char)* 3 * width * height);
    fread(img, sizeof(unsigned char), height * width * 3, input_file);
    
    //open output image
    FILE *output_file = fopen("output.ppm", "wb");
    fprintf(output_file, "P6\n%d\n%d\n%d\n", width, height, MAX_COLOR_VALUE);
    
    //open histogram file
    FILE *output_txt = fopen("output.txt", "w");
    int* histogram = (int*)calloc(6, sizeof(int));
    
    unsigned char *cur = img;
    unsigned char *new_cur = new_img;
    
    //create new image
    
    //first line
    for (int i = 0; i < width; ++i){
        for (int j = 0; j < 3; ++j) {
            *(new_cur++) = *(cur++);
        }
        histogram[(int)(round(0.2126* (*(new_cur-3)) + 0.7152*(*(new_cur-2)) + 0.0722* (*(new_cur-1))) / 51)]++;
    }
    
    //center
    width *= 3;
    int max_ind = (height - 2) * width;
    for(int i = 0; i < max_ind; i+=3) {
        if (i % width == 0 || (i + 3) % width == 0) {
            *(new_cur++) = *(cur++);
            *(new_cur++) = *(cur++);
            *(new_cur++) = *(cur++);
        }
        else {
            int value = -1 * (*(cur - width)) - (*(cur - 3)) + 5 * (*cur) - (*(cur + 3)) - (*(cur + width));
            value = (value < 0) ? 0 : (value > 255) ? 255 : value;
            *(new_cur++) = (unsigned char)value;
            cur++;
            
            value = -1 * (*(cur - width)) - (*(cur - 3)) + 5 * (*cur) - (*(cur + 3)) - (*(cur + width));
            value = (value < 0) ? 0 : (value > 255) ? 255 : value;
            *(new_cur++) = (unsigned char)value;
            cur++;
            
            value = -1 * (*(cur - width)) - (*(cur - 3)) + 5 * (*cur) - (*(cur + 3)) - (*(cur + width));
            value = (value < 0) ? 0 : (value > 255) ? 255 : value;
            *(new_cur++) = (unsigned char)value;
            cur++;
        }
        histogram[(int)(round(0.2126* (*(new_cur-3)) + 0.7152*(*(new_cur-2)) + 0.0722* (*(new_cur-1))) / 51)]++;
    }
    width /= 3;
    
    //last line
    for (int i = 0; i < width; ++i){
        for (int j = 0; j < 3; ++j) {
            *(new_cur++) = *(cur++);
        }
        histogram[(int)(round(0.2126* (*(new_cur-3)) + 0.7152*(*(new_cur-2)) + 0.0722* (*(new_cur-1))) / 51)]++;
    }
    
    //write output image
    fwrite(new_img, 1, height * width * 3, output_file);
    
    //write histogram
    fprintf(output_txt, "%d %d %d %d %d", histogram[0], histogram[1], histogram[2], histogram[3], histogram[4]+histogram[5]);
    
    free(histogram);
    free(img);
    free(new_img);
    
    fclose(input_file);
    fclose(output_file);
    fclose(output_txt);
    
    return 0;
}

