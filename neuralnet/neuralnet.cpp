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

/**
** Function to adjust weights of the net
** curr_layer -> Size of the current layer of neurons to work with.
** size_curr -> Number of neurons in the current layer.
** prev_layer -> layer of neurons previous to the current one. NULL if working with input layer.
** size_prev -> size of the previous layer
** next_layer -> layer of neurons directly after the current one. NULL if working with output layer,
** size_next -> size of the next layer.
** d -> array of the expected output. Can be NULL if working with inner/input layers.
** input -> Current input to the net. NULL if the not input layer
*/
void adjustWeights(double alpha, neuron **curr_layer, int size_curr, neuron **prev_layer, int size_prev, \
                   neuron **next_layer, int size_next, double *d = NULL, double *input = NULL)
{
    if (!curr_layer)
    {
        std::cout << "No current layer available" << std::endl;
        return;
    }

    // Delta, output, weight to be adjusted of the current neuron, and the delta of the current weight
    double delta, y, w, dw;
    int w_size;                 // Number of weights of the current neuron
    // Output of the previous neuron
    double x;
    // Deltas next neuron and weights connecting the current with the next
    double ndelta, wij;
    int i, j;
    double sum;
    // Start iterating over the current layer
    for (i = 0; i < size_curr; i++)
    {
        delta = 0;
        sum = 0;
        // Calculate delta of current neuron
        y = curr_layer[i]->getOutput();
        //When dealing with output layer
        if (!next_layer)
        {
            delta = y * (1 - y) * (d[i] - y);
        }
        // When dealing with an inner/input layer
        else
        {
            delta = y * (1 - y);
            for (j = 0; j < size_next; j++)
            {
                ndelta = next_layer[j]->getDelta();
                wij = next_layer[j]->getIthWeight(i);
                sum += (ndelta * wij);
            }
            delta *= sum;
        }
        // Update delta of the current neuron
        curr_layer[i]->setDelta(delta);
        w_size = curr_layer[i]->get_Wsize();
        for (j = 0; j < w_size; j++)
        {
            // Get weight to update
            w = curr_layer[i]->getIthWeight(j);
            // Get output of the previous layer
            if (!prev_layer)                        // Previous layer exists
                x = prev_layer[j]->getOutput();
            else
                x = input[j];
            dw = alpha * delta * x;
            // Update weight
            w += dw;
            curr_layer[i]->setIthWeight(w, j);
        }

    }
}

/**
** Implementation of the backpropagation algorithm
** x -> Inputs.
** d -> expected output
*/
void neuralnet::backpropagation(double *x, double *d)
{
    if (!x || !d)
    {
        std::cout << "NULL arrays" << std::endl;
        return;
    }

    int i;
    /// Propagate inputs through the net
    feedforward(x);

    // Adjust weight for the output layer
    adjustWeights(alpha, &output, nn_output, &hidden_layers[nh_layer - 1], nn_hlayer[nh_layer - 1], NULL, 0, d, NULL);

    // Adjust for the hidden layers
    for (i = nh_layer - 1; i >= 0; i++)
    {
        neuron *next, *prev;
        int size_next, size_prev;
        if (i == nh_layer - 1)
        {
            next = output;
            size_next = nn_output;
        }
        else
        {
            next = hidden_layers[i + 1];
            size_next = nn_hlayer[i + 1];
        }
        if (i == 0)
        {
            prev = input;
            size_prev = nn_input;
        }
        else
        {
            prev = hidden_layers[i - 1];
            size_prev = nn_hlayer[i - 1];
        }
        adjustWeights(alpha, &hidden_layers[i], nn_hlayer[i], &prev, size_prev, &next, size_next, NULL, NULL);
    }

    // Adjust weights of the input layer
    adjustWeights(alpha, &input, nn_input, NULL, 0, &hidden_layers[0], nn_hlayer[0], NULL, x);
}

/**
** Function to train the net.
** x -> Inputs to train the net.
** d -> Expected outputs of the net}
** totalIter -> Number of total iterations to train the net
** sizeX -> size of the inputs.
*/
void neuralnet::train(double **x, double **d, int sizeX, int totalIter)
{
    int i, numIter;

    cout << "Training network" << endl;
    numIter = 0;
    while (numIter < totalIter)
    {
        for (i = 0; i < sizeX; i++)
            backpropagation(x[i], d[i]);
    }

    cout << "Train complete" << endl;
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
            delete[] hidden_layers[i];
        delete[] hidden_layers;
    }
}
