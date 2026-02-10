# BASIC NEURAL NETWORK IN C

```bash
cmake . -B build
cmake --build build -j 8
```

## how does it work tho

statistical model. takes inputs and does matrix shit

when training, it compares the output values against "expected" values and determines a "cost." this
can be used to differentiate a "gradient" and use the chain rule to perform "gradient descent"

```
scratchpad

dc/da_f = d/dy[C(x, y)] where x is expected and y is the output of the matrix
da_f/dz_f = d/dz[A(z)]
dc/dz_f = dc/da_f * da_f/dz_f

```
