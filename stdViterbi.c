//Standard Viterbi Based on A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition
//Online Viterbi based on The on-line Viterbi algorithm (Master’s Thesis) by Rastislav Šrámek

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <unistd.h> // For waiting time on Linux


const int K = 4; //# of Hidden States
const int M = 4; //# of Observation Symbols
const int T = 100; //# of time instances

const double A[K][K] = { { 0.96, 0.04, 0.0, 0.0 }, //Transition Probability Matrix
                           { 0, 0.95, 0.05, 0.0 }, 
                           { 0.0, 0.0, 0.85, 0.15 }, 
                           { 0.1, 0.0, 0.0, 0.9 } };

const double E[K][M] =  {{0.6, 0.2, 0.0, 0.2},    //Emission Matrix
                        {0.1, 0.8, 0.1, 0.0},  
                        {0.0, 0.14, 0.76, 0.1},    
                        {0.1, 0.0, 0.1, 0.8}};    

const double initial[K] =  { 0.25, 0.25, 0.25, 0.25};  //Initial distribution

/////////////////////////// Online viterbi data structures ////////////////////

typedef struct node {
    int children;  //# of children
    int t;        //time instance 
    int j;        //state
    struct node * parent;   //Parent node on tree
    struct node * previous;   //Previous node on linked list
} node_t;

typedef struct pcolumn {
   double col[K];            //Scores column
   struct pcolumn * next;    
   struct pcolumn * previous;
} prob_column_t;

typedef struct scolumn {
   int col[K];            //Column for indexes that maximize probabilites
   struct scolumn * next;
   struct scolumn * previous;
} state_column_t;

int delta_t = 0;         //Time delta from leaves to root
node_t * last = NULL;    //Last node of linked list
node_t * root = NULL;    //Root of the tree
node_t * leaves[K] = {NULL};  //List of leaves

prob_column_t * probs_first_column = NULL;  //Pointer to first column of probabilities matrix
prob_column_t * probs_last_column = NULL;   //Pointer to last column of probabilities matrix
state_column_t * states_first_column = NULL;   //Pointer to first column of states matrix
state_column_t * states_last_column = NULL;   //Pointer to last column of states matrix

 
void online_viterbi_initialization(int starting_state){
   last = NULL;
   //Create scores and path matrices
   prob_column_t * p_first = NULL;
   p_first = (prob_column_t *) calloc(1, sizeof(prob_column_t));
   p_first->next = NULL; 
   p_first->previous = NULL;

   state_column_t * s_first = NULL;
   s_first = (state_column_t *) calloc(1, sizeof(state_column_t));
   s_first->next = NULL; 
   s_first->previous = NULL;

   for (int j = 0; j < K; j++){
      p_first->col[j] = initial[j];
      s_first->col[j] = starting_state;
   }
   //p_first->col[starting_state] = 1;

   probs_first_column = p_first;
   probs_last_column = p_first;

   states_first_column = s_first;
   states_last_column = s_first;

   //Create leaves
   for(int j = 0; j<K; j++){
         node_t * new_node = NULL;
         new_node = (node_t *) calloc(1, sizeof(node_t));
         new_node->j = j;
         new_node->t = 0;
         new_node->parent = NULL;
         new_node->previous = last;
         new_node->children = 0;
         last = new_node;
         leaves[j] = new_node;
   }
   root = leaves[starting_state];   //Store initial root
}


void compress(int t) {
   node_t * current = last;
   node_t * aux = NULL;   //Auxiliary node
   node_t * last_visited = last;   //Auxiliary node

   while (current != NULL) {
      if(current->children == 0 && current->t != t) {
         if(current->parent != NULL){
            current->parent->children -=1;
            aux = current->previous;
            last_visited->previous = current->previous;
            free(current);
            current = aux;
         }        
      } else {
         while(current->parent != NULL && current->parent->children == 1){
            aux = current->parent->parent;
            current->parent = aux;
         }
      }
      if(current->previous == current){
         current->previous = NULL;
      }
      last_visited = current;
      current = current->previous;
   }

   //Free remaininig nodes with 0 children that are not leaves
   current = last;
   last_visited = last;
   while (current != NULL) {
      if(current->children == 0 && current->t != t) {
         aux = current->previous;
         last_visited->previous = current->previous;
         free(current);
         current = aux;  
      }
      last_visited = current;
      if(current != NULL){current = current->previous;}
      
   }




}

bool find_new_root(){ 
   //Returns true if root has changed based on time delta between previous root and new root

   delta_t = 0;
   node_t * current = last;
   node_t * aux = last;   //Auxiliary node - Last node that has 2 or more children
   delta_t = last->t;

   /*
   Find last node that has at least 2 children starting from any leave or node with at least two children

   Proof and analysis of this algorithm can be found on 
   "Šrámek R., Brejová B., Vinař T. (2007) On-Line Viterbi Algorithm for Analysis of Long Biological Sequences. 
   In: Giancarlo R., Hannenhalli S. (eds) Algorithms in Bioinformatics. WABI 2007. 
   Lecture Notes in Computer Science, vol 4645. Springer, Berlin, Heidelberg"
  */
  
   while (current != NULL) {
      if (current->children >= 2){
         aux = current;
      }
      current = current->parent;
   }
   if(aux != root){  //Test if root has changed
      delta_t -= aux->t;
      if(delta_t == 0){
         return false;
      }
      else
      {
         root = aux;
         return true;
      }
   }
   return false;
}

void traceback(){
   prob_column_t * p_col = probs_last_column;
   prob_column_t * P = NULL;
   state_column_t * s_col = states_last_column;
   state_column_t * S = NULL;     //Auxiliary state column

   int output = root->j;
   printf("%d ", output);
      for (int i = 0; i < delta_t; i++)  {
          if(s_col != NULL){    //Find column corresponding to root
            s_col = s_col->previous;
          }
          if(p_col != NULL){    //Find column corresponding to root
            p_col = p_col->previous;         
         }
        
   }
   
   if (s_col != NULL && s_col->next != NULL){s_col->next->previous = NULL;}
   if ( p_col != NULL &&  p_col->next != NULL){ p_col->next->previous = NULL;}

   while(p_col != NULL && p_col->previous != NULL && s_col->previous!= s_col){    //Traceback from new root to previous root
      output = s_col->col[output];
      printf("%d ", output);
      S = s_col;
      s_col = s_col->previous;
      s_col->next = NULL;
      P = p_col;
      p_col = p_col->previous;
      p_col->next = NULL;
      S = NULL;
      P = NULL;
   } 
   printf("\n");

}

void traceback_last_part(){
   prob_column_t * p_col = probs_last_column;
   prob_column_t * P = NULL;
   state_column_t * s_col = states_last_column;
   state_column_t * S = NULL;     //Auxiliary state column

   int output = root->j;
   printf("%d ", output);
   
   if (s_col != NULL && s_col->next != NULL){s_col->next->previous = NULL;}
   if (p_col != NULL &&  p_col->next != NULL){ p_col->next->previous = NULL;}


   while(p_col != NULL && p_col->previous != NULL && s_col->previous!= s_col){    //Traceback from new root to previous root
      output = s_col->col[output];
      printf("%d ", output);
      S = s_col;
      s_col = s_col->previous;
      s_col->next = NULL;
      P = p_col;
      p_col = p_col->previous;
      p_col->next = NULL;
      S = NULL;
      P = NULL;
   } 
   printf("\n");

}

void update(int t, int observation){
   /*
      Online Viterbi algorithm : Updates scores and paths (max_idexes) matrices when the observation at time t in received
      observation: observation at time t
   */

   //Create new matrices columns
   prob_column_t * p_new = NULL;
   p_new = (prob_column_t *) calloc(1, sizeof(prob_column_t));
   
   p_new->previous = probs_last_column;
   probs_last_column->next = p_new;
   probs_last_column = p_new;

   state_column_t * s_new = NULL;
   s_new = (state_column_t *) calloc(1, sizeof(state_column_t));
 
   s_new->previous = states_last_column;
   states_last_column->next = s_new;
   states_last_column = s_new;

   for (int j = 0; j < K; j++){
      double max = -1;
      double aux = -1;
      int max_index = 0;
      for (int i = 0; i < K; i++){
         aux = probs_last_column->previous->col[i]*A[i][j];  
         if (aux > max){
            max = aux*E[j][observation];
            max_index = i;
         }
      }
      //Store score and path
      p_new->col[j] = max;
      s_new->col[j] = max_index;

      //Add new leaf nodes
      node_t * new_node = NULL;
      new_node = (node_t *) calloc(1, sizeof(node_t));
      new_node->t = t;
      new_node->j = j;
      new_node->children = 0;
      new_node->parent = leaves[max_index];
      new_node->parent->children +=1;
      new_node->previous = last;
      last = new_node;
      //leaves[j] = new_node;
   }
   leaves[3] = last;
   leaves[2] = last->previous;
   leaves[1] = last->previous->previous;
   leaves[0] = last->previous->previous->previous; 

   compress(t);

   if(find_new_root()){
      traceback();
   }  
}

void printList(){
   node_t *aux = last;
   while(aux != NULL){
      printf("t: %d,  j: %d \n", aux->t, aux->j);
      aux = aux->previous;
   }
   printf("\n\n");
}

///////////////////////////////////////////////////////////////////

double scores[K][T] = {{0}};  //delta_{t}(i)
int path[K][T] = {{0}};     //Argument that maximized scores
int optimalPath[T] = {0};


void std_viterbi_initialization(int* observations){
   for (size_t i = 0; i < M; i++){
      scores[i][0] = initial[i]*E[i][observations[0]];
   }
}

void std_viterbi_recursion(int* observations) {
   double max = -1;
   double aux = -1;
   int max_index = 0;
   for (int t = 1; t < T; t++){
      for (int j = 0; j < K; j++){
         max = -1;
         aux = -1;
         max_index = 0;
         for (int i = 0; i < K; i++){
            aux = scores[i][t-1]*A[i][j];
            if (aux > max){
               max = aux;
               max_index = i;
            }
         }
         scores[j][t] = max*E[j][observations[t]];
         path[j][t] = max_index;
      }
   }
}

void std_viterbi_termination(){
   int max_index = 0;
   double max = 0;
   for (int j = 0; j < K; j++){
      if(scores[j][T-1] > max){
         max = scores[j][T-1];
         max_index = j;
      }
   }
   optimalPath[T-1] = max_index;

   for (int t = T - 2; t >=0 ; t--){
      optimalPath[t] = path[optimalPath[t + 1]][t + 1];
   }
}

void std_viterbi(int* observations){
   std_viterbi_initialization(observations);
   std_viterbi_recursion(observations);
   std_viterbi_termination();
}

void printArray(int size, int *array){
   for(int j = 0; j < size; j++) {
        printf("%d ", array[j]);
    }
}

int main() {
  int previous = 0;
  static int observations[T] = {0};
  
   //int observations[T] = {1,2,3,0,1,2,3,0,1,2,3,3,3,3,3,3,0,2,1,2};
   //std_viterbi(observations);
   //printArray(T, optimalPath);

   int count = 0;
   for(int i = 0; i<100*60*60; i++){
     
      count = i%T;
      observations[count] = (previous + rand()%2)%4;
      previous = observations[count];
      if(count == 0){
         online_viterbi_initialization(observations[count]);
      } else {
         update(i, observations[count]);
      }

      if(count == T-1){

         traceback_last_part();

         printf("\nobservations:  ");
         printArray(T, observations);
         std_viterbi(observations);
         printf("\nStd Viterbi window:  ");
         printArray(T, optimalPath);
         printf("\n\n");
      }
      usleep(100000);
   } 

  
   
   /*
   //printf("\n\n");
   online_viterbi_initialization(1);

   for (int t = 1; t < 20; t++) {
      update(t, observations[t]);
      printList();
   }
   
   
   //update(0, 1);
   */


   return 0;
}  