#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
        int distance;
        int height;
        int width;
    } note_t;

int main(){
    //input
    int door_width, notes_count;
    scanf("%d %d", &door_width, &notes_count);

    note_t* notes = (note_t*)malloc(notes_count * sizeof(note_t));
    for (int i = 0; i < notes_count; i++){
        scanf("%d %d %d", &(notes[i].distance), &(notes[i].height), &(notes[i].width));
    }

    //
    int* max_heights = (int*)calloc(door_width, sizeof(int));
    bool is_visible[notes_count];

    for (int i = 0; i < notes_count; i++){
        is_visible[i] = false;
    }
    
    //
    for (int i = 0; i < notes_count; i++){
        bool flag = false;
        for (int j = notes[i].distance; j < notes[i].distance + notes[i].width; j++){
            if (notes[i].height > max_heights[j]){
                max_heights[j] = notes[i].height;
                flag = true;
            }
        }
        if (flag)
                is_visible[i] = true;
    }

    //
    for (int i = 0; i < door_width; i++){
        max_heights[i] = 0;
    }

    //
    int two_side_visible = 0;
    int one_side_visible = 0;
    for (int i = notes_count - 1; i >= 0; i--){
        bool flag = false;
        for (int j = notes[i].distance; j < notes[i].distance + notes[i].width; j++){
            if (notes[i].height > max_heights[j]){
                max_heights[j] = notes[i].height;
                flag = true;
            }
        }
        if (flag && is_visible[i])
            two_side_visible++;
        else if (flag || is_visible[i])
            one_side_visible++;
    }

    printf("%d %d %d", two_side_visible, one_side_visible, notes_count-one_side_visible-two_side_visible);

    free(max_heights);
    free(notes);
    return 0;
}