#include <iostream>
#include "neurone.h"
#include "ActFuncs.h"

using namespace std;

void printInputs(double *x, int s){
    cout << "Inputs: ";
    for (int i = 0; i < s; i++)
        cout << x[i] << "\t";
    cout << endl;
}

int main()
{
    int s = 3;
    int i;
    double x[s] = {3.1, -1.5, 0.5};
    double w[s] = {1.1, 0.2, -2.1};
    double bias = 1.0, y;
    neuron n;

    n.setWeightSize(s);
    n.setWeights(w);
    n.setBias(bias);
    n.setActFunction(stepAF);

    /*printInputs(x, s);
    neuron n;
    n.compute(x);
    printInputs(x, s);
    cout << n << endl;*/

    n.initializeRandomWeights();
    for (i = 0; i < 1000; i++) {
        n.compute(x);
        y = n.getOutput();
    }

    return 0;
}
