#include "neuralnet.h"

neuralnet::neuralnet()
{
    nn_input = 0;
    nn_output = 0;
    nh_layer = 0;
    nn_hlayer = NULL;

    input = NULL;
    output = NULL;
    hidden_layers = NULL;
}

neuralnet::neuralnet(int n_input, int n_output, int n_layer, int *hidden)
{
    int i;
    nn_input = n_input;
    nn_output = n_output;
    nh_layer = n_layer;
    nn_hlayer = new int[nh_layer];

    for (i = 0; i < nh_layer; i++)
        nn_hlayer[i] = hidden[i];

    /// Initialize array or neurons
    input = new neuron[nn_input];
    output = new neuron[nn_output];
    hidden_layers = new neuron*[nh_layer];
    for (i = 0; i < nh_layer; i++)
        hidden_layers[i] = new neuron[nn_hlayer[i]];

    /// Initialize output's array
    net_output = new double[nn_output];
}

void neuralnet::init(double a, double (*actFunct)(double))
{
    int i, j;
    alpha = a;

    /// Initialize input layer
    for (i = 0; i < nn_input; i++)
    {
        input[i].setWeightSize(nn_input);
        input[i].initializeRandomWeights();
        input[i].setActFunction(actFunct);
    }

    /// Initialize hidden layers

    /// Initialize first hidden layer
    for (i = 0; i < nn_hlayer[0]; i++)
    {
        hidden_layers[0][i].setWeightSize(nn_input);
        hidden_layers[0][i].initializeRandomWeights();
        hidden_layers[0][i].setActFunction(actFunct);
    }

    /// Initialize the rest
    for (i = 1; i < nh_layer; i++)
        for (j = 0; j < nn_hlayer[i]; j++)
        {
            hidden_layers[i][j].setWeightSize(nn_hlayer[i - 1]);
            hidden_layers[i][j].initializeRandomWeights();
            hidden_layers[i][j].setActFunction(actFunct);
        }

    /// Initialize output layer
    for (i = 0; i < nn_output; i++)
    {
        output[i].setWeightSize(nn_hlayer[nh_layer - 1]);
        output[i].initializeRandomWeights();
        output[i].setActFunction(actFunct);
    }
}

void neuralnet::feedforward(double *x)
{
    double *o, *oaux;
    int i, j;

    o = new double[nn_input];
    for (i = 0; i < nn_input; i++)
    {
        input[i].compute(x);
        o[i] = input[i].getOutput();
    }

    for (i = 0; i < nh_layer; i++)
    {
        oaux = new double[nn_hlayer[i]];
        for (j = 0; j < nn_hlayer[i]; j++)
        {
            hidden_layers[i][j].compute(o);
            oaux[i] = hidden_layers[i][j].getOutput();
        }
        delete[] o;
        o = new double[nn_hlayer[i]];
        for (j = 0; j < nn_hlayer[i]; j++)
            o[j] = oaux[j];
        delete[] oaux;
    }

    for (i = 0; i < nn_output; i++)
    {
        output[i].compute(o);
        net_output[i] = output[i].getOutput();
    }
}

void neuralnet::getOutput(double **out)
{
    if (!out)
        return;

    for (int i = 0; i < nn_output; i++)
        *out[i] = net_output[i];
}

neuralnet::~neuralnet()
{
    if (input)
        delete[] input;
    if (output)
        delete[] output;
    if (hidden_layers)
    {
        for (int i = 0; i < nh_layer; i++)
            delete[] hidden_layer[i];
        delete[] hidden_layer;
    }
}
