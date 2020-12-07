#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <unistd.h>

int final_diff = 0;
typedef struct node{
  int diff;
  int abs_diff; 
  struct node *left; 
  struct node *right; 
} node_t;

node_t* new_node(int abs_diff, int diff){
  node_t* node = (node_t*)malloc(sizeof(node_t)); 
  node->diff = diff;
  node->abs_diff = abs(diff);
  node->left = NULL;
  node->right = NULL;
  return(node);
}

node_t* build(int *model_weight, int start, int end, int last_sum){
    int best_border = 0;
    int min_diff = 0, min_diff_abs = 1e9;
    int best_sum_right = 0, best_sum_left = 0;

    int sum_left = 0;
    int sum_right = last_sum;
    for (int border = start + 1; border < end; border++){
        sum_left += model_weight[border - 1];
        sum_right -= model_weight[border - 1];
        
        int cur_diff = sum_left - sum_right;
        int cur_diff_abs = abs(cur_diff);

        if (cur_diff_abs < min_diff_abs){
            min_diff_abs = cur_diff_abs;
            min_diff = cur_diff;
            best_border = border;
            best_sum_right = sum_right;
            best_sum_left = sum_left;
        }
    }
    final_diff += min_diff_abs;
    node_t *node = new_node(min_diff_abs, min_diff); 
    if (best_border - start > 1)
        node->left = build(model_weight, start, best_border, best_sum_left);
    if (end - best_border > 1)
        node->right = build(model_weight, best_border, end, best_sum_right);
    return (node);
}

int foo(node_t* node, int pilot_weight, bool two_pilots){
    if (node == NULL)
        return 0;
    int sum = INT_MAX;
    if (two_pilots){
        if (node->diff < 0)
            sum = abs(node->diff + 2 * pilot_weight) - node->abs_diff + foo(node->left, pilot_weight, true);
        else
            sum = abs(node->diff - 2 * pilot_weight) - node->abs_diff + foo(node->right, pilot_weight, true);
        int sum_left = foo(node->left, pilot_weight, false);
        int sum_right = foo(node->right, pilot_weight, false);
        if (sum_left + sum_right > sum)
            return sum;
        return sum_left + sum_right;
    } else {
        int cur_diff_if_left = abs(node->diff + pilot_weight);
        int cur_diff_if_right = abs(node->diff - pilot_weight);
        int sum_left = cur_diff_if_left - node->abs_diff + foo(node->left, pilot_weight, false);
        int sum_right = cur_diff_if_right - node->abs_diff + foo(node->right, pilot_weight, false);
        if (sum_left < sum_right)
            return sum_left;
        return sum_right;
    }
}


int main(){
    int models_number;
    int pilot_weight;
    int models_sum = 0;
    scanf("%d %d", &models_number, &pilot_weight);

    int* model_weight = (int*)malloc(models_number * sizeof(int));
    for (int i = 0; i < models_number; i++){
        scanf("%d", &model_weight[i]);
        models_sum += model_weight[i];
    }

    node_t* root = build(model_weight, 0, models_number, models_sum);
    printf("%d ", final_diff);

    int sum = foo(root, pilot_weight, true);
    printf("%d\n", final_diff + sum);

    free(model_weight);
    return 0;
}
