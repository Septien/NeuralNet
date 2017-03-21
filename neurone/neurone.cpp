#include "neurone.h"
#include <time.h>

neuron::neuron() {
    bias = 0;
    weights = NULL;
    output = 0;
    actFunction = NULL;
    w_size = 0;
    delta = 0;
}

neuron::neuron(double *w, int n) {
    w_size = n;
    weights = new double[w_size];
    bias = 0;
    actFunction = NULL;
    output = 0;
    delta = 0;
}

neuron::neuron(double *w, int n, double b)  {
    w_size = n;
    weights = new double[w_size];
    bias = b;
    actFunction = NULL;
    output = 0;
    delta = 0;
}

neuron::neuron(double *w, int n, double b, double (*actFunc)(double))  {
    w_size = n;
    weights = new double[w_size];
    bias = b;
    actFunction = actFunc;
    output = 0;
    delta = 0;
}

void neuron::setWeights(double *w) {
    if (weights)
        delete weights;

    weights = new double[w_size];
    for (int i = 0; i < w_size; i++)
        weights[i] = w[i];
}

void neuron::setIthWeight(double w, int i)
{
    if (i < 0 || i >= w_size)
    {
        std::cout << "Index out of bounds" << std::endl;
        return;
    }
    weights[i] = w;
}

void neuron::getWeights(double **w) {
    int i;
    if (!w)
        return;
    for (i = 0; i < w_size; i++)
        *w[i] = weights[i];
}

void neuron::initializeRandomWeights() {
    srand((unsigned int)time(NULL));
    int i;
    for (i = 0; i < w_size; i++)
        weights[i] = (((double)rand()) / ((double)RAND_MAX)) * 1;
    /// Init bias
    bias = (((double)rand()) / ((double)RAND_MAX)) * 1;
}

double neuron::get_Wsize()
{
    return w_size;
}

double neuron::getIthWeight(int i)
{
    if (i < 0 || i >= w_size)
    {
        std::cout << "Index out of range" << std::endl;
        return -1;
    }

    return weights[0];
}

void neuron::setDelta(double d)
{
    delta = d;
}

double neuron::getDelta()
{
    return delta;
}

void neuron::setWeightSize(int n) {
    w_size = n;
    if (!weights)
        weights = new double[w_size];
}

int neuron::getWeightSize() {
    return w_size;
}

void neuron::setBias(double b) {
    bias = b;
}

double neuron::getBias() {
    return bias;
}

void neuron::setActFunction(double (*actFunc)(double)) {
    actFunction = actFunc;
}

double neuron::getOutput() {
    return output;
}

void neuron::compute(double *x) {
    double y = 0;
    int i;

    for (i = 0; i < w_size; i++)
        y += x[i] * weights[i];

    y += bias;
    output = actFunction(y);
}


/**Function to train the neuron.
** x -> matrix containing the input samples.
** y -> array containing the expected value.
*/
void neuron::train(double **x, double *y) {
    int i;
    srand(time(NULL));
    for (i = 0; i < w_size; i++)
        weights[i] = rand();
}

std::ostream& operator<<(std::ostream& out, const neuron &n) {
    int i;
    out << "Weights: ";
    for (i = 0; i < n.w_size; i++)
        out << n.weights[i] << "\t";
    out << "\n";
    out << "Bias: " << n.bias << "\n";
    out << "Output: " << n.output << "\n";

    return out;
}

neuron::~neuron() {
    if (weights)
        delete weights;
    actFunction = NULL;
}

/**
** Initialize a neuron with the given values
*/
void initNeuron(neuron &n, int weightSize, double (*actFunct)(double))
{
    n.setWeightSize(weightSize);
    n.initializeRandomWeights();
    n.setActFunction(actFunct);
}
