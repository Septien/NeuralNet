#ifndef NEURONE_H_INCLUDED
#define NEURONE_H_INCLUDED

#include <iostream>
#include <stdlib.h>

class neuron {
private:
    int w_size;
    double *weights;
    double bias;
    double output;
    double delta;
    //Activation function
    double (*actFunction)(double);

public:
    neuron();
    neuron(double *, int);
    neuron(double *, int, double);
    neuron(double *, int, double, double (*actFunc)(double));

    void setWeights(double *);
    void setIthWeight(double, int);
    void getWeights(const double **);
    double getIthWeight(int);
    void initializeRandomWeights();
    double get_Wsize();
    int getWeightSize();

    void setDelta(double);
    double getDelta();

    void setWeightSize(int);

    void setBias(double);
    double getBias();

    double getOutput();

    void setActFunction(double (*actFunc)(double));

    void compute(double *);
    void train(double **, double *);

    friend std::ostream& operator<<(std::ostream&, const neuron&);

    ~neuron();
};

#endif // NEURONE_H_INCLUDED
