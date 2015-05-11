# Introduction
CNN is a toolkit in C++ for specifying and working with neural networks. Specifically,
the goal of the toolkit to allow straightforward definitions of neural architectures
in the form of symbolic computational graphs. Nodes in this graph, also called the 
symbolic objects are typically scalars, vectors or tensors representing input, parameters
or internal states of the neural network. Operations on nodes in this graph typically
belong to a small set (concatenation, non-linear functions, semiring operations, softmax, etc.).
CNN makes creating nodes and operations on the hypergraph fairly straightforward. Through this
example, we will explore some of the basic features provided by CNN. In the course of this tutorial,
we may also explore some advanced topics and suggest topics further reading and exploration.

# XOR with a Multi-layer Perceptron
The example we will work with is a simple feed-forward neural network
([the multi-layer perceptron](http://en.wikipedia.org/wiki/Feedforward_neural_network#Multi-layer_perceptron))
which can implement the XOR operation. The complete code for this example is at examples/xor.cc. If you
have already built CNN, a binary for this example can be found at build/examples/xor and can be run.

## The structure of a neural network specification with CNN
The general workflow of implementing a neural network would be the following : 
```cpp
// 1. Include CNN headers

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  
  // 2. Define and initialize hyperparameters
  
  // 3. Define model, inputs and parameters

  // 4. Build the symbolic (hyper)graph

  //    4.1 Add symbolic nodes

  //    4.2 Define operations on nodes

  // 5. Training parameters
}
```
### CNN headers and initialization
Three header files and the cnn namespace contain the bulk of the functionality available through CNN.
We start by including these.
```cpp
#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
...

using namespace cnn;

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
```

```Initialize()``` initializes a scratch space for computation. In the future, this may be used for better ways of
initializing memory for the model parameters and the computation graph.

### Defining hyperparameters
Hyperparamters are distinct from the parameters of the model and in the case of neural networks,
describe things like the model structure (eg. number of nodes in the hidden state) or parameters
for the learning algorithm (eg. learning rate, batch size). We will use two of these in our
implementation of the XOR network.

```cpp
// hyper-parameters
const unsigned HIDDEN_SIZE = 8;
const unsigned ITERATIONS = 30;
```
HIDDEN_SIZE is the size of the hidden layer and ITERATIONS is the maximum
number of iterations for which SGD (the parameter learning) algorithm will
run.

### Initializing model and its parameters
Our goal is to learn the parameters (vectors, tensors) of a model given some data the we observe.
We initialize a model and malloc memory for its parameters. For the model, we
also specify a learning algorithm (SGD in this case)

```cpp
Model m;
SimpleSGDTrainer sgd(&m);

Parameters& p_a = *m.add_parameters({1});
Parameters& p_b1 = *m.add_parameters({HIDDEN_SIZE/2});
Parameters& p_b2 = *m.add_parameters({HIDDEN_SIZE/2});
Parameters& p_W = *m.add_parameters({HIDDEN_SIZE, 2});
Parameters& p_V = *m.add_parameters({1, HIDDEN_SIZE});
```
A couple of things are worth elaboration. First, the "Model" keeps track
of the parameters (and by extension their location in memory). CNN uses
this information and passes pointers to these memory locations to Eigen
when necessary. It also provides functions such as `load` and `save` for
parameters to perform disk writing and reading.

`add_parameters` is used to add parameters to the model. It takes
the size of the parameter as an argument. Examples:
* add_parameters({N}) : Vector of size N
* add_parameters({N, 2}) : Matrix of size Nx2
* add_parameters({1, N}) : Column vector of size 1xN

### Initializing the computational graph

```cpp
#include "cnn/edges.h"
#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cnn;

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  // parameters
  const unsigned HIDDEN_SIZE = 8;
  const unsigned ITERATIONS = 30;
  Model m;
  SimpleSGDTrainer sgd(&m);
  //MomentumSGDTrainer sgd(&m);

  Parameters& p_a = *m.add_parameters({1});
  Parameters& p_b1 = *m.add_parameters({HIDDEN_SIZE/2});
  Parameters& p_b2 = *m.add_parameters({HIDDEN_SIZE/2});
  Parameters& p_W = *m.add_parameters({HIDDEN_SIZE, 2});
  Parameters& p_V = *m.add_parameters({1, HIDDEN_SIZE});

  // build the graph
  Hypergraph hg;

  // get symbolic variables corresponding to parameters
  VariableIndex i_b1 = hg.add_parameter(&p_b1);
  VariableIndex i_b2 = hg.add_parameter(&p_b2);
  VariableIndex i_b = hg.add_function<Concatenate>({i_b1, i_b2});
  VariableIndex i_a = hg.add_parameter(&p_a);
  VariableIndex i_W = hg.add_parameter(&p_W);
  VariableIndex i_V = hg.add_parameter(&p_V);

  vector<float> x_values(2);  // set x_values to change the inputs to the network
  VariableIndex i_x = hg.add_input({2}, &x_values);
  cnn::real y_value;  // set y_value to change the target output
  VariableIndex i_y = hg.add_input(&y_value);

  // two options: MatrixMultiply and Sum, or Multilinear
  // these are identical, but Multilinear may be slightly more efficient
#if 1
  VariableIndex i_f = hg.add_function<MatrixMultiply>({i_W, i_x});
  VariableIndex i_g = hg.add_function<Sum>({i_f, i_b});
#else
  VariableIndex i_g = hg.add_function<Multilinear>({i_b, i_W, i_x});
#endif
  VariableIndex i_h = hg.add_function<Tanh>({i_g});

#if 0
  VariableIndex i_p = hg.add_function<MatrixMultiply>({i_V, i_h});
  VariableIndex i_y_pred = hg.add_function<Sum>({i_p, i_a});
#else
  VariableIndex i_y_pred = hg.add_function<Multilinear>({i_a, i_V, i_h});
#endif
  hg.add_function<SquaredEuclideanDistance>({i_y_pred, i_y});
  hg.PrintGraphviz();
  if (argc == 2) {
    ifstream in(argv[1]);
    boost::archive::text_iarchive ia(in);
    ia >> m;
  }

  // train the parameters
  for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
    double loss = 0;
    for (unsigned mi = 0; mi < 4; ++mi) {
      bool x1 = mi % 2;
      bool x2 = (mi / 2) % 2;
      x_values[0] = x1 ? 1 : -1;
      x_values[1] = x2 ? 1 : -1;
      y_value = (x1 != x2) ? 1 : -1;
      loss += as_scalar(hg.forward());
      hg.backward();
      sgd.update(1.0);
    }
    sgd.update_epoch();
    loss /= 4;
    cerr << "E = " << loss << endl;
  }
  boost::archive::text_oarchive oa(cout);
  oa << m;
}
```
