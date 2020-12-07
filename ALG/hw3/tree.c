
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



#include <stdio.h>
#include <stdlib.h>

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

node_t* build(int *model_weight, int start, int end){
    int best_border = 0, min_diff_abs = 1e9;
    int min_diff = 0;

    for (int border = 1; border < end; border++){
        int sum_left = 0, sum_right = 0;
        for (int i = start; i < border; i++)
            sum_left += model_weight[i];
        for (int i = border; i < end; i++)
            sum_right += model_weight[i];
        
        int cur_diff = sum_left - sum_right;
        int cur_diff_abs = abs(cur_diff);

        if (cur_diff_abs < min_diff_abs){
            min_diff_abs = cur_diff_abs;
            min_diff = cur_diff;
            best_border = border;
        }
    }
    // printf("best borders %d\n", best_border);
    //printf("min diff = %d, best border = %d\n", min_diff_abs, best_border);
    final_diff += min_diff_abs;
    node_t *node = new_node(min_diff_abs, min_diff); 
    if (best_border - start > 1)
        node->left = build(model_weight, start, best_border);
    if (end - best_border > 1)
        node->right = build(model_weight, best_border, end);
    return (node);
}

/*
int foo(node_t* node, int *new_diff, int pilot_weight){
    int new_diff_left = 0, new_diff_right = 0;

    if (node == NULL){
        *new_diff = 0;
        return 0;
    }
    int sum_left = foo(node->left, &new_diff_left, pilot_weight);
    int sum_right = foo(node->right, &new_diff_right, pilot_weight);
    int sum = node->abs_diff + sum_left + sum_right;
    printf("new_diff_left = %d, new_diff_right = %d\n", new_diff_left, new_diff_right);
    printf("sum_left = %d, sum_right = %d\n", sum_left, sum_right);
    //printf("node's abs diff = %d\n", node->abs_diff);
    
    int tmp_diff_left = new_diff_left + sum_right + abs(node->diff + pilot_weight);
    int tmp_diff_right = new_diff_right + sum_left + abs(node->diff - pilot_weight);
    printf("tmp_diff_left = %d, tmp_diff_right = %d\n", tmp_diff_left, tmp_diff_right);
    *new_diff = min(tmp_diff_left, tmp_diff_right);
    printf("new_diff = %d\n", *new_diff);


    printf("\n");
    
    return sum;
}
*/

int foo(node_t* node, int pilot_weight){
    if (node == NULL)
        return 0;
    int sum = 0;
    int cur_diff_if_left = abs(node->diff + pilot_weight);
    int cur_diff_if_right = abs(node->diff - pilot_weight);
    int sum_left = cur_diff_if_left - node->abs_diff + foo(node->left, pilot_weight);
    int sum_right = cur_diff_if_right - node->abs_diff + foo(node->right, pilot_weight);

    if (sum_left < sum_right){
        sum = sum_left;
    } else {
        sum = sum_right;
    }

    // if (cur_diff_if_left < cur_diff_if_right){
    //     sum = cur_diff_if_left - node->abs_diff + foo(node->left, pilot_weight);
    //     // printf("sum if left = %d\n", sum);
    // } else {
    //     sum = cur_diff_if_right - node->abs_diff + foo(node->right, pilot_weight);
    //     // printf("sum if right = %d\n", sum);
    // }
    return sum;
}


int main(){
    int models_number;
    int pilot_weight;
    scanf("%d %d", &models_number, &pilot_weight);

    int* model_weight = (int*)malloc(models_number * sizeof(int));
    for (int i = 0; i < models_number; i++)
        scanf("%d", &model_weight[i]);

    node_t* root = build(model_weight, 0, models_number);
    int sum_left = foo(root->left, pilot_weight);
    int sum_right = foo(root->right, pilot_weight);
    

    // int tmp = -1;
    // printf("            %d\n", root->abs_diff);
    // printf("    %d             %d  \n", root->left->abs_diff, root->right->abs_diff);
    // printf("  %d  %d         %d    %d\n", root->left->left->abs_diff, root->left->right->abs_diff, root->right->left->abs_diff, root->right->right->abs_diff);
    // printf(" %d %d %d %d    %d %d %d %d\n", root->left->left->left->abs_diff, tmp,
    //                                         tmp, root->left->right->right->abs_diff, 
    //                                         tmp, root->right->left->right->abs_diff, 
    //                                         root->right->right->left->abs_diff, root->right->right->right->abs_diff);
    // for (int i = 0; i < models_number; i++)
    //     printf("%d ", model_weight[i]);
    // printf("\n\n");

    printf("%d %d\n", final_diff, final_diff + sum_left + sum_right);

    // int sum = foo(root, pilot_weight);
    // printf("%d %d\n", final_diff, final_diff + sum);
    free(model_weight);
    return 0;
}