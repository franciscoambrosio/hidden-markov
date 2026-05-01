# Hidden Markov Models — Viterbi Decoding in C

Fast C implementations of two Viterbi algorithms for decoding Hidden Markov Models:

1. **Standard Viterbi**: classic two-pass algorithm (initialization → recursion → traceback)
2. **Online Viterbi**: streaming variant with path compression for near-real-time decoding

Both decode a sequence of T observations into the most likely sequence of hidden states.

## Build

```bash
make          # compile to 'viterbi' binary
make run      # compile and run the test
make clean    # remove binary
```

Or manually:
```bash
gcc -O2 -Wall -o viterbi stdViterbi.c -lm
./viterbi
```

## Test

The program runs a **correctness test** that:
- Generates a random observation sequence (T=1000, with fixed seed for reproducibility)
- Decodes it with Online Viterbi (streaming, one observation at a time)
- Decodes it with Standard Viterbi (full sequence at once)
- Compares: if paths match, **PASS**; if they differ, **FAIL**

```
$ make run

HMM Viterbi — correctness test
  States: 4  |  Symbols: 4  |  Sequence length: 1000

PASS  both algorithms produce identical paths (1000 states)
```

Exit code 0 = PASS, non-zero = FAIL.

## HMM Parameters

- **States**: 4 (transition matrix A)
- **Symbols**: 4 (emission matrix E)
- **Initial distribution**: uniform [0.25, 0.25, 0.25, 0.25]

Transition and emission matrices are hard-coded in `stdViterbi.c` for this test.

## References

- **Standard Viterbi**: Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition."
- **Online Viterbi**: Šrámek, R. (2007). "The On-Line Viterbi Algorithm" (Master's Thesis).

## Implementation Notes

The **Online Viterbi** uses a tree-based path compression strategy:
- Maintains a tree of all possible paths as observations arrive
- When paths merge (share a common ancestor), the earlier part is committed
- Compresses single-child chains to save memory
- Emits decoded states as they become certain

This allows streaming decoding without buffering the entire sequence, useful for long or infinite observation streams.
