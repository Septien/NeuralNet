#ifndef NEURALNET_H_INCLUDED
#define NEURALNET_H_INCLUDED

#include "../neurone/neurone.h"

class neuralnet {
private:
    neuron *input;
    neuron *output;
    neuron **hidden_layers;

    int nn_input;               //Size of input layer
    int nn_output;              //Size of output layer
    int *nn_hlayer;             //Array containing the number of neurons of each layer
    int nh_layer;               //Number of hidden layers
    /// Learning rate
    double alpha;

    /// Output of the net
    double *net_output;

public:
    neuralnet();
    neuralnet(int n_input, int n_output, int n_layer, int *hidden);
    void init(double, double (*actFunc)(double));
    void feedforward(double *);
    void backpropagation(double *);
    void getOutput(double **);
    ~neuralnet();
};


#endif // NEURALNET_H_INCLUDED
