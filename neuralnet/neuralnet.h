#ifndef NEURALNET_H_INCLUDED
#define NEURALNET_H_INCLUDED

#include <fstream>
#include "../neurone/neurone.h"
using namespace std;

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

    void backpropagation(double *, double *);

public:
    neuralnet();
    neuralnet(int n_input, int n_output, int n_layer, int *hidden);
    void init(double, double (*actFunc)(double));
    void feedforward(double *);
    void getOutput(double **);
    void saveNettoFile(char *);
    void loadNetfromFile(char *);
    void train(double **, double **, int, int);
    ~neuralnet();
};

void adjustWeights(double, neuron **, int , neuron **, int, neuron **, int, double *, double *);
void saveWeight(neuron *, int, int, ofstream &);

#endif // NEURALNET_H_INCLUDED
