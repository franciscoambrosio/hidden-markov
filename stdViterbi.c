//Based on A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition

#include <stdio.h>

const int K = 4; //# of Hidden States
const int M = 4; //# of Observation Symbols
const int T = 16; //# of time instances

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


void initialization(int* observations){
   for (size_t i = 0; i < M; i++){
      scores[i][0] = initial[i]*E[i][observations[0]];
   }
}

void recursion(int* observations){
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

void termination(){
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

void viterbi(int* observations){
   initialization(observations);
   recursion(observations);
   termination();
}

void printArray(int size, int *array){
   for(int j = 0; j < size; j++) {
        printf("%d ", array[j]);
    }
}

int main() {
   int observations[T] = {0,0,1,1,1,2,3,2,2,3,3,3,3,0,3,3};
   viterbi(observations);
   printArray(T, optimalPath);
   return 0;
}  