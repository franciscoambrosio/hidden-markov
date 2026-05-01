// Standard Viterbi based on "A Tutorial on Hidden Markov Models and
// Selected Applications in Speech Recognition" by Rabiner (1989).
// Online Viterbi based on "The On-Line Viterbi Algorithm" (Master's
// Thesis) by Rastislav Šrámek.

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define K (4) // number of hidden states
#define M (4) // number of observation symbols
#define T (1000) // sequence length

const double A[K][K] = {
    {0.96, 0.04, 0.0,  0.0 },
    {0.0,  0.95, 0.05, 0.0 },
    {0.0,  0.0,  0.85, 0.15},
    {0.1,  0.0,  0.0,  0.9 },
}; // transition probability matrix

const double E[K][M] = {
    {0.6,  0.2,  0.0,  0.2},
    {0.1,  0.8,  0.1,  0.0},
    {0.0,  0.14, 0.76, 0.1},
    {0.1,  0.0,  0.1,  0.8},
}; // emission matrix

const double initial[K] = {0.25, 0.25, 0.25, 0.25}; // initial distribution

// ─────────────────────────────────────────────────────────────────────────────
// Shared utilities
// ─────────────────────────────────────────────────────────────────────────────

static const double LOG_ZERO = -2000000.0; // lower bound for log probabilities

static double bounded_log(double a)
{
    return (a == 0.0) ? LOG_ZERO : log(a);
}

static double log_sum2(double a, double b)
{
    double s = a + b;
    return (s < LOG_ZERO) ? LOG_ZERO : s;
}

static double log_sum3(double a, double b, double c)
{
    double s = a + b + c;
    return (s < LOG_ZERO) ? LOG_ZERO : s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Standard Viterbi
// ─────────────────────────────────────────────────────────────────────────────

static double scores[K][T]; // delta_t(i): best log-probability ending at state i, time t
static int    path[K][T];   // argmax that achieved scores[i][t]
static int    optimalPath[T];

static void std_viterbi(int *observations)
{
    // Initialization
    for (int j = 0; j < K; j++) {
        double max = LOG_ZERO;
        int    max_index = 0;
        for (int i = 0; i < K; i++) {
            double v = log_sum2(bounded_log(initial[i]), bounded_log(A[i][j]));
            if (v > max) { max = v; max_index = i; }
        }
        scores[j][0] = log_sum2(max, bounded_log(E[j][observations[0]]));
        path[j][0]   = max_index;
    }

    // Recursion
    for (int t = 1; t < T; t++) {
        for (int j = 0; j < K; j++) {
            double max = LOG_ZERO;
            int    max_index = 0;
            for (int i = 0; i < K; i++) {
                double v = log_sum2(scores[i][t - 1], bounded_log(A[i][j]));
                if (v > max) { max = v; max_index = i; }
            }
            scores[j][t] = log_sum2(max, bounded_log(E[j][observations[t]]));
            path[j][t]   = max_index;
        }
    }

    // Termination + traceback
    int max_index = 0;
    double max = LOG_ZERO;
    for (int j = 0; j < K; j++) {
        if (scores[j][T - 1] > max) { max = scores[j][T - 1]; max_index = j; }
    }
    optimalPath[T - 1] = max_index;
    for (int t = T - 2; t >= 0; t--)
        optimalPath[t] = path[optimalPath[t + 1]][t + 1];
}

// ─────────────────────────────────────────────────────────────────────────────
// Online Viterbi — tree-based path compression
// ─────────────────────────────────────────────────────────────────────────────

typedef struct node {
    int          children;
    int          t;
    int          j;
    struct node *parent;
    struct node *previous; // linked list through all nodes
} node_t;

typedef struct pcolumn {
    double          col[K];
    struct pcolumn *next;
    struct pcolumn *previous;
} prob_column_t;

typedef struct scolumn {
    int             col[K];
    struct scolumn *next;
    struct scolumn *previous;
} state_column_t;

static int          delta_t = 0;
static node_t      *last    = NULL;
static node_t      *root    = NULL;
static int          prev_root_time = -1;
static node_t      *leaves[K];

static prob_column_t  *probs_first_column  = NULL;
static prob_column_t  *probs_last_column   = NULL;
static state_column_t *states_first_column = NULL;
static state_column_t *states_last_column  = NULL;

static int decoded_stream[T];
static int decoded_stream_idx = 0;

static void online_viterbi_init(void)
{
    last              = NULL;
    decoded_stream_idx = 0;
    root              = NULL;
    prev_root_time    = -1;

    prob_column_t *p = calloc(1, sizeof(prob_column_t));
    state_column_t *s = calloc(1, sizeof(state_column_t));

    for (int j = 0; j < K; j++) {
        p->col[j] = bounded_log(initial[j]);
        s->col[j] = 0;
        leaves[j] = NULL;
    }

    probs_first_column  = probs_last_column  = p;
    states_first_column = states_last_column = s;
}

static void compress(int t)
{
    node_t *current     = last;
    node_t *last_visited = last;
    node_t *aux;

    // Path compression: merge single-child chains, prune dead leaves
    while (current != NULL) {
        if (current->children == 0 && current->t != t) {
            if (current->parent != NULL) {
                current->parent->children -= 1;
                aux = current->previous;
                last_visited->previous = current->previous;
                free(current);
                current = aux;
                last_visited = current;
                if (current != NULL) current = current->previous;
            } else {
                last_visited = current;
                current = current->previous;
            }
        } else {
            while (current->parent != NULL && current->parent->children == 1)
                current->parent = current->parent->parent;
            last_visited = current;
            current = current->previous;
        }
    }

    // Second pass: free remaining zero-child non-leaf nodes
    current     = last;
    last_visited = last;
    while (current != NULL) {
        if (current->children == 0 && current->t != t) {
            aux = current->previous;
            if (current->parent)
                current->parent->children -= 1;
            last_visited->previous = current->previous;
            free(current);
            current = aux;
            last_visited = current;
            if (current != NULL) current = current->previous;
        } else {
            last_visited = current;
            if (current != NULL) current = current->previous;
        }
    }
}

static void free_all_nodes(void)
{
    node_t *current = last;
    while (current != NULL) {
        node_t *temp = current->previous;
        free(current);
        current = temp;
    }
    last = NULL;
}

static bool find_new_root(void)
{
    delta_t = 0;

    // Wait until all paths share a common ancestor
    if (root == NULL) {
        node_t *leaf = last;
        node_t *traced_root[K];
        for (int i = 0; i < K; i++) {
            node_t *track = leaf;
            node_t *temp  = NULL;
            while (track != NULL) { temp = track; track = track->parent; }
            traced_root[i] = temp;
            leaf = leaf->previous;
        }
        for (int i = 1; i < K; i++)
            if (traced_root[0] != traced_root[i]) return false;
    }

    // Walk the chain from the deepest leaf to find the node with the most
    // recent branching point — that is the new root to decode up to.
    node_t *current = last;
    node_t *aux     = NULL;
    delta_t = last->t;

    while (current != NULL) {
        if (current->children >= 2) aux = current;
        current = current->parent;
    }

    if (aux != NULL && aux != root) {
        delta_t -= aux->t;
        if (delta_t == 0) return false;
        if (root) prev_root_time = root->t;
        root = aux;
        return true;
    }
    return false;
}

static void traceback(void)
{
    prob_column_t  *p_col = probs_last_column;
    state_column_t *s_col = states_last_column;
    int interim[T];
    int interim_idx = 0;

    int output = root->j;
    interim[interim_idx++] = output;

    for (int i = 0; i < delta_t; i++) {
        s_col = s_col->previous;
        p_col = p_col->previous;
    }
    if (s_col->next) s_col->next->previous = NULL;
    if (p_col->next) p_col->next->previous = NULL;

    int depth = (prev_root_time == -1) ? root->t : (root->t - prev_root_time - 1);
    while (depth-- > 0) {
        output = s_col->col[output];
        interim[interim_idx++] = output;
        state_column_t *S = s_col; s_col = s_col->previous; free(S);
        prob_column_t  *P = p_col; p_col = p_col->previous; free(P);
    }

    while (p_col != NULL) {
        state_column_t *S = s_col; s_col = s_col->previous; free(S);
        prob_column_t  *P = p_col; p_col = p_col->previous; free(P);
    }

    for (int i = interim_idx - 1; i >= 0; i--)
        decoded_stream[decoded_stream_idx++] = interim[i];
}

static void traceback_last_part(void)
{
    prob_column_t  *p_col = probs_last_column;
    state_column_t *s_col = states_last_column;
    int interim[T];
    int interim_idx = 0;

    double max = LOG_ZERO;
    int    max_index = 0;
    for (int i = 0; i < K; i++) {
        if (p_col->col[i] > max) { max = p_col->col[i]; max_index = i; }
    }

    int output = max_index;
    interim[interim_idx++] = output;

    if (s_col->next) s_col->next->previous = NULL;
    if (p_col->next) p_col->next->previous = NULL;

    int depth = (root == NULL) ? (T - 1) : (T - 1 - root->t - 1);
    while (depth-- > 0) {
        output = s_col->col[output];
        interim[interim_idx++] = output;
        state_column_t *S = s_col; s_col = s_col->previous; free(S);
        prob_column_t  *P = p_col; p_col = p_col->previous; free(P);
    }

    while (p_col != NULL) {
        state_column_t *S = s_col; s_col = s_col->previous; free(S);
        prob_column_t  *P = p_col; p_col = p_col->previous; free(P);
    }

    for (int i = interim_idx - 1; i >= 0; i--)
        decoded_stream[decoded_stream_idx++] = interim[i];
}

static void update(int t, int observation)
{
    prob_column_t *p_new  = calloc(1, sizeof(prob_column_t));
    state_column_t *s_new = calloc(1, sizeof(state_column_t));

    p_new->previous = probs_last_column;
    probs_last_column->next = p_new;
    probs_last_column = p_new;

    s_new->previous = states_last_column;
    states_last_column->next = s_new;
    states_last_column = s_new;

    for (int j = 0; j < K; j++) {
        double max = LOG_ZERO;
        int    max_index = 0;
        for (int i = 0; i < K; i++) {
            double v = log_sum3(
                probs_last_column->previous->col[i],
                bounded_log(A[i][j]),
                bounded_log(E[j][observation])
            );
            if (v > max) { max = v; max_index = i; }
        }
        p_new->col[j] = max;
        s_new->col[j] = max_index;

        node_t *new_node  = calloc(1, sizeof(node_t));
        new_node->t       = t;
        new_node->j       = j;
        new_node->parent  = leaves[max_index];
        if (new_node->parent) new_node->parent->children += 1;
        new_node->previous = last;
        last = new_node;
    }

    // Reattach leaves array to the K nodes just added
    node_t *cur = last;
    for (int j = K - 1; j >= 0; j--) {
        leaves[j] = cur;
        cur = cur->previous;
    }

    compress(t);

    if (find_new_root())
        traceback();
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: compare both algorithms on a random observation sequence
// ─────────────────────────────────────────────────────────────────────────────

int main(void)
{
    static int observations[T];

    // Fixed seed — results are reproducible across runs
    srand(42);

    printf("HMM Viterbi — correctness test\n");
    printf("  States: %d  |  Symbols: %d  |  Sequence length: %d\n\n", K, M, T);

    // Generate a plausible observation sequence: each symbol is the same as
    // or adjacent (mod M) to the previous one, mimicking correlated HMM output.
    int prev = 0;
    for (int i = 0; i < T; i++) {
        observations[i] = (prev + rand() % 2) % M;
        prev = observations[i];
    }

    // Run Online Viterbi: feed observations one at a time (streaming)
    online_viterbi_init();
    for (int i = 0; i < T; i++)
        update(i, observations[i]);
    traceback_last_part();

    // Run Standard Viterbi: needs the full sequence upfront
    std_viterbi(observations);

    // Compare paths
    int mismatches = 0;
    for (int i = 0; i < T; i++)
        if (optimalPath[i] != decoded_stream[i]) mismatches++;

    if (mismatches == 0)
        printf("PASS  both algorithms produce identical paths (%d states)\n", T);
    else
        printf("FAIL  %d / %d states differ\n", mismatches, T);

    free_all_nodes();
    return mismatches > 0;
}
