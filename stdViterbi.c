//Standard Viterbi Based on A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition
//Online Viterbi based on The on-line Viterbi algorithm (Master’s Thesis) by Rastislav Šrámek

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>
#include <unistd.h> // For waiting time on Linux
#include <time.h> // For waiting time on Linux
#include <math.h>


#define  K  (4)          //# of Hidden States
#define  M  (4)          //# of Observation Symbols
#define  T  (1000)        //# of time instances

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
int prev_root_time = -1;
node_t * leaves[K] = {NULL};  //List of leaves

prob_column_t * probs_first_column = NULL;  //Pointer to first column of probabilities matrix
prob_column_t * probs_last_column = NULL;   //Pointer to last column of probabilities matrix
state_column_t * states_first_column = NULL;   //Pointer to first column of states matrix
state_column_t * states_last_column = NULL;   //Pointer to last column of states matrix

int decoded_stream[T];
int decoded_stream_idx=0;

double B = -2000000;       // lower bound for log probabilities

double bounded_log(double a) {
    if (a == 0)
        return B;
    else
        return log(a);
}

double bounded_log_sum( int num, ... ) {

    va_list arguments;                     
    double sum = 0;

    va_start ( arguments, num );           

    for ( int x = 0; x < num; x++ )        
        sum +=  va_arg ( arguments, double ) ; 

    va_end ( arguments );
 
    if (sum < B)
      sum = B;

    return sum;
}


void online_viterbi_initialization(int starting_state){
   last = NULL;
   decoded_stream_idx = 0;

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
      p_first->col[j] = bounded_log(initial[j]);
      s_first->col[j] = starting_state;
   }
   //p_first->col[starting_state] = 1;

   probs_first_column = p_first;
   probs_last_column = p_first;

   states_first_column = s_first;
   states_last_column = s_first;

   //Create leaves
   for(int j = 0; j<K; j++){
         leaves[j] = NULL;
   }

   root = NULL;
   prev_root_time = -1;
}


void compress(int t) {
   node_t * current = (node_t *)  last;
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
            last_visited = current;
            current = current->previous;            
         } else {
            last_visited = current;
            current = current->previous;     
         }

      } else {
         while(current->parent != NULL && current->parent->children == 1){
            aux = current->parent->parent;
            current->parent = aux;
         }
         last_visited = current;
         current = current->previous;              
      }
      // if(current->previous == current){
      //    current->previous = NULL;
      // }

   }

   //Free remaininig nodes with 0 children that are not leaves
   current = last;
   last_visited = last;
   while (current != NULL) {
      if(current->children == 0 && current->t != t) {
         aux = current->previous;
         if (current->parent) current->parent->children -= 1;
         last_visited->previous = current->previous;
         free(current);
         current = aux;  

         last_visited = current;
         if(current != NULL){current = current->previous;}

      }
      else {
         last_visited = current;
         if(current != NULL){current = current->previous;}
      }
      // last_visited = current;
      // if(current != NULL){current = current->previous;}
      
   }
}


void free_all_nodes(void) {
   node_t * current = (node_t *) last;
   node_t * temp = NULL;

   while (current != NULL) {

      temp = current->previous;
      free(current);
      current = temp;
      
   }
}


bool find_new_root(){ 
   //Returns true if root has changed based on time delta between previous root and new root

   delta_t = 0;
   node_t * current = (node_t *) last;
   node_t * aux = NULL;   //Auxiliary node - Last node that has 2 or more children

   // first make sure path has merged
   if (root==NULL){
      node_t *leaf = last;
      node_t *traced_root[K] = {NULL,};
      node_t *temp = NULL;
      node_t *track = NULL;
      for (int i=0 ; i<K ; i++){
         track = leaf;
         while (track != NULL){
            temp =  track;
            track = track->parent;
            if (track==NULL)
               traced_root[i] = temp;
         }
         leaf = leaf->previous;
      }

      bool merged = true;
      for (int i=1;i<K;i++){
         if (traced_root[0] != traced_root[i]) {
            merged = false;
            break;
         }
      }

      if (!merged)
         return false;
   }

   /*
   Find last node that has at least 2 children starting from any leave or node with at least two children

   Proof and analysis of this algorithm can be found on 
   "Šrámek R., Brejová B., Vinař T. (2007) On-Line Viterbi Algorithm for Analysis of Long Biological Sequences. 
   In: Giancarlo R., Hannenhalli S. (eds) Algorithms in Bioinformatics. WABI 2007. 
   Lecture Notes in Computer Science, vol 4645. Springer, Berlin, Heidelberg"
  */
  

   current = (node_t *) last;
   delta_t = last->t;

   while (current != NULL) {
      if (current->children >= 2){
         aux = current;
      }
      current = current->parent;
   }
   if (aux !=NULL)
      if(aux != root){  //Test if root has changed
         delta_t -= aux->t;
         if(delta_t == 0){
            return false;
         }
         else
         {
            if (root)
               prev_root_time = root->t;
            root = aux;
            // printf("\n New root found (time : %d, state : %d)", root->t, root->j);
            return true;
         }
      }
   else
      return false;
}

void traceback(){
   prob_column_t * p_col = probs_last_column;
   prob_column_t * P = NULL;
   state_column_t * s_col = states_last_column;
   state_column_t * S = NULL;     //Auxiliary state column
   int interim_decoded_stream[T] = {0,};
   int interim_decoded_stream_idx = 0;
   int depth=0;

   int output = root->j;
   printf("%d ", output);
   interim_decoded_stream[interim_decoded_stream_idx++] = output;

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

   if (prev_root_time == -1)
      depth = root->t;
   else
      depth = (root->t - prev_root_time - 1);

   while(depth-- > 0){    //Traceback from new root to previous root
      output = s_col->col[output];
      printf("%d ", output);
      interim_decoded_stream[interim_decoded_stream_idx++] = output;
      S = s_col;
      s_col = s_col->previous;
      // s_col->next = NULL;
      P = p_col;
      p_col = p_col->previous;
      // p_col->next = NULL;
      free(S);
      free(P);
      S = NULL;
      P = NULL;
   } 
   printf("\n");


   // free further remaining 
   while (p_col!=NULL){
      S = s_col;
      s_col = s_col->previous;
      // s_col->next = NULL;
      P = p_col;
      p_col = p_col->previous;
      // p_col->next = NULL;
      free(S);
      free(P);
      S = NULL;
      P = NULL;
   }


   for(int i=interim_decoded_stream_idx-1;i>=0;i--)
      decoded_stream[decoded_stream_idx++] = interim_decoded_stream[i];

}

void traceback_last_part(){
   prob_column_t * p_col = probs_last_column;
   prob_column_t * P = NULL;
   state_column_t * s_col = states_last_column;
   state_column_t * S = NULL;     //Auxiliary state column
   int interim_decoded_stream[T] = {0,};
   int interim_decoded_stream_idx = 0;
   int depth = 0;

   // get index for maximum value of last column
   double max = B;
   double aux = B;
   int max_index = 0;
   for (int i = 0; i < K; i++){
      aux = p_col->col[i];
      if (aux > max){
         max = aux;
         max_index = i;
      }
   }

   int output = max_index;
   printf("%d ", output);
   interim_decoded_stream[interim_decoded_stream_idx++] = output;

   if (s_col != NULL && s_col->next != NULL){s_col->next->previous = NULL;}
   if (p_col != NULL &&  p_col->next != NULL){ p_col->next->previous = NULL;}

   if (root==NULL)
      depth = (T-1);
   else
      depth = (T-1) - root->t - 1;

   while(depth-- > 0){    //Traceback from new root to previous root
      output = s_col->col[output];
      printf("%d ", output);
      interim_decoded_stream[interim_decoded_stream_idx++] = output;

      S = s_col;
      s_col = s_col->previous;
      // s_col->next = NULL;
      P = p_col;
      p_col = p_col->previous;
      // p_col->next = NULL;
      free(S);
      free(P);
      S = NULL;
      P = NULL;
   } 
   printf("\n");

   // free further remaining 
   while (p_col!=NULL){
      S = s_col;
      s_col = s_col->previous;
      // s_col->next = NULL;
      P = p_col;
      p_col = p_col->previous;
      // p_col->next = NULL;
      free(S);
      free(P);
      S = NULL;
      P = NULL;
   }

   for(int i=interim_decoded_stream_idx-1;i>=0;i--)
      decoded_stream[decoded_stream_idx++] = interim_decoded_stream[i];

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
   p_new->next = NULL;
   probs_last_column->next = p_new;
   probs_last_column = p_new;

   state_column_t * s_new = NULL;
   s_new = (state_column_t *) calloc(1, sizeof(state_column_t));
 
   s_new->previous = states_last_column;
   s_new->next = NULL;
   states_last_column->next = s_new;
   states_last_column = s_new;

   for (int j = 0; j < K; j++){
      double max = B;
      double aux = B;
      int max_index = 0;
      for (int i = 0; i < K; i++){
         aux = bounded_log_sum( 3, probs_last_column->previous->col[i], bounded_log(A[i][j]), bounded_log(E[j][observation]) );
         if (aux > max){
            max = aux;
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
      if (new_node->parent)
         new_node->parent->children +=1;
      new_node->previous = last;
      last = new_node;
      //leaves[j] = new_node;
   }

   node_t * current = last;
   for (int j = K; j > 0 ; j--){
      leaves[j-1] = current;
      current = current->previous;
   }

   // printf("\nBefore compress\n");
   // printList();
   // printProbList();
   // printStateList();

   compress(t);

   // printf("\nAfter compress\n");
   // printList();

   if(find_new_root()){
      // printf("Found New root : (%d, %d) \n", root->t, root->j);
      traceback();
   }
}

void printList(){
   node_t *aux = last;
   while(aux != NULL){
      printf("t: %d,  j: %d , parent: (%d, %d), child: %d \n", aux->t, aux->j, (aux->parent!=NULL)? aux->parent->t : -1, (aux->parent!=NULL)? aux->parent->j: -1, aux->children );
      aux = aux->previous;
   }
   printf("\n\n");
}


void printProbList(){
   prob_column_t * p_col = probs_last_column;   //Pointer to last column of probabilities matrix

   printf("\n");
   while(p_col!=NULL){
      for(int i=0;i<K;i++) {
         printf("%0.3lf  ", p_col->col[i]);
      }
      printf("\n");
      p_col = p_col->previous;
   }
}



void printStateList(){
   state_column_t * s_col = states_last_column;   //Pointer to last column of probabilities matrix

   printf("\n");
   while(s_col!=NULL){
      for(int i=0;i<K;i++) {
         printf("%d  ", s_col->col[i]);
      }
      printf("\n");
      s_col = s_col->previous;
   }
}


///////////////////////////////////////////////////////////////////

double scores[K][T] = {{0}};  //delta_{t}(i)
int path[K][T] = {{0}};     //Argument that maximized scores
int optimalPath[T] = {0};


void std_viterbi_initialization(int* observations){
   double max = B;
   double aux = B;
   int max_index = 0;
   for (size_t j = 0; j < K; j++){

      max = B;
      aux = B;
      max_index = 0;
      for (int i = 0; i < K; i++){
         aux = bounded_log_sum( 2, bounded_log(initial[i]), bounded_log(A[i][j]) );
         if (aux > max){
            max = aux;
            max_index = i;
         }
      }
      scores[j][0] = bounded_log_sum( 2, max, bounded_log(E[j][observations[0]]) );
      path[j][0] = max_index;
   }
}

void std_viterbi_recursion(int* observations) {
   double max = B;
   double aux = B;
   int max_index = 0;
   for (int t = 1; t < T; t++){
      for (int j = 0; j < K; j++){
         max = B;
         aux = B;
         max_index = 0;
         for (int i = 0; i < K; i++){
            aux = bounded_log_sum( 2, scores[i][t-1], bounded_log(A[i][j]) );
            if (aux > max){
               max = aux;
               max_index = i;
            }
         }
         scores[j][t] = bounded_log_sum( 2, max, bounded_log(E[j][observations[t]]) );
         path[j][t] = max_index;
      }
   }
}

void std_viterbi_termination(){
   int max_index = 0;
   double max = B;
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
   int count = 0;

   srand((unsigned)time(NULL));

   online_viterbi_initialization(0);

   for(int i = 0; i<100*60*60 ;i++){
     
      count = i%T;
      observations[count] = (previous + rand()%2)%4;
      previous = observations[count];

      update(count, observations[count]);

      if(count == T-1){

         traceback_last_part();

         printf("\nobservations:  ");
         printArray(T, observations);
         std_viterbi(observations);
         printf("\nStd Viterbi window:  ");
         printArray(T, optimalPath);
         printf("\nOnline Viterbi window:  ");
         printArray(T, decoded_stream);
         printf("\n\n");

         // check if two path matches
         int j;
         for (j=0;j<T;j++) {
            if ( optimalPath[j] != decoded_stream[j] )
               break;
         }
         if (j==T)
            printf("Results match! \n");
         else
            printf("Results don't match\n ");

         usleep(1000000);

         // for fresh new start of online viterbi decoding
         free_all_nodes();
         online_viterbi_initialization(0);
      }

      usleep(10000);
   } 

   return 0;
}  

