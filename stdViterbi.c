//Standard Viterbi Based on A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition
//Online Viterbi based on The on-line Viterbi algorithm (Master’s Thesis) by Rastislav Šrámek


#include <stdio.h>
const int K = 4; //# of Hidden States
const int M = 4; //# of Observation Symbols
const int T = 16; //# of time instances
///////////////////////////

typedef struct node {
    int children;  //# of children
    int t;        
    int j;
    struct node * parent;
    struct node * previous;
} node_t;

typedef struct pcolumn {
   double col[K];
   struct pcolumn * next;
   struct pcolumn * previous;
} prob_column_t;

typedef struct scolumn {
   int col[K];
   struct scolumn * next;
   struct scolumn * previous;
} state_column_t;

node_t * last = NULL;
node_t * leaves[K] = {NULL};  //List of pointers to leaves
prob_column_t * probs_first_column = NULL;  //Pointer to first column of probabilities matrix
prob_column_t * probs_last_column = NULL;   //Pointer to last column of probabilities matrix
state_column_t * states_first_column = NULL;   //Pointer to first column of states matrix
state_column_t * states_last_column = NULL;   //Pointer to last column of states matrix
const int MAX_WINDOW_SIZE = 10;

void online_viterbi_initialization(){
   prob_column_t * p_first = NULL;
   p_first = (prob_column_t *) malloc(sizeof(prob_column_t));
   p_first->next = NULL; p_first->previous = NULL;
   state_column_t * s_first = NULL;
   s_first = (state_column_t *) malloc(sizeof(state_column_t));
   s_first->next = NULL; s_first->previous = NULL;

   for (int j = 0; j < K; j++){
      p_first->col[j] = initial[j];
      s_first->col[j] = -1;
   }

   probs_first_column = p_first;
   probs_last_column = p_first;

   states_first_column = s_first;
   states_last_column = s_first;

   node_t * new = NULL;
   new = (node_t *) malloc(sizeof(node_t));
   for(int j = 0; j<K; j++){
         node_t * new = NULL;
         new = (node_t *) malloc(sizeof(node_t));
         new->j = j;
         new->t = 0;
         new->parent = NULL;
         new->previous = last;
         last = new;
         leaves[j] = new;
   }
}

void update(int t, int observation){
   /*
      Online Viterbi algorithm : Updates probabilities and states=path matrices 
      when the observation at time t in received
      int observation: X_{j} = observation at time t

   */
   //Create new matrices columns
   prob_column_t * p_new = NULL;
   p_new = (prob_column_t *) malloc(sizeof(prob_column_t));
   p_new->next = NULL; p_new->previous = probs_last_column;

   state_column_t * s_new = NULL;
   s_new = (state_column_t *) malloc(sizeof(state_column_t));
   s_new->next = NULL; s_new->previous = states_last_column;

   for (size_t j = 0; j < K; j++){
      double max = -1;
      double aux = -1;
      int max_index = 0;
      for (size_t i = 0; i < K; i++){
         aux = probs_last_column->col[j]*A[i][j];  //!!!!!!!!
         if (aux > max){
            max = aux;
            max_index = i;
         }
      }
      //Store score and path
      p_new->col[j] = max*E[j][observation];
      s_new->col[j] = max_index;

      //Add new leaf nodes
      node_t * new = NULL;
      new = (node_t *) malloc(sizeof(node_t));
      new->t = t;
      new->j = j;
      new->children = 0;
      new->parent = leaves[max_index];
      new->parent->children +=1;
      new->previous = last;
      last = new;
   }
   
   
   
}

void compress(int t) {
   node_t * current = last;
   node_t * T = NULL;   //Auxiliary node
   while (current->previous != NULL) {
      if(current->children < 1 && current->t != t) {
         current->parent->children -=1;
         T = current->previous;
         free(current);
         current = T;
      } else {
         while(current->parent->children == 1){
            T = current->parent->parent;
            free(current->parent);
            current->parent = T;
         }
      }
      current = current->previous;
   }
}

///////////////////////////////////////////////////////////////////





const double A[K][K] = { { 0.8, 0.2, 0.0, 0.0 }, //Transition Probability Matrix
                           { 0, 0.4, 0.6, 0.0 }, 
                           { 0.0, 0.0, 0.7, 0.3 }, 
                           { 0.1, 0.0, 0.0, 0.9 } };

const double E[K][M] =  {{0.9, 0.0, 0.05, 0.05},    //Emission Matrix
                        {0.1, 0.8, 0.1, 0.0},   //Emission Matrix
                        {0, 0.0, 0.9, 0.1},    //Emission Matrix
                        {0.2, 0.0, 0.05, 0.75}};    //Emission Matrix

const double initial[K] =  { 0.25, 0.25, 0.25, 0.25};  //Initial distribution

double scores[K][T] = {{0}};  //Quantity delta_{t}(i)
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
   for (size_t t = 1; t < T; t++){
      for (size_t j = 0; j < K; j++){
         max = -1;
         aux = -1;
         max_index = 0;
         for (size_t i = 0; i < K; i++){
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
   for (size_t j = 0; j < K; j++){
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
   int observations[T] = {0,0,1,1,1,2,3,2,2,3,3,3,3,0,3,3};
   std_viterbi(observations);
   printArray(T, optimalPath);
   return 0;
}  