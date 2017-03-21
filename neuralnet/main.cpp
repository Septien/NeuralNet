#include <iostream>
#include "neuralnet.h"
#include "../neurone/ActFuncs.h"

using namespace std;

int main()
{
    int hidden[] = {4};
    double *x = new double[3];
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;

    neuralnet nnt(3, 2, 1, hidden);
    nnt.init(0.02, sigmoidAF);
    nnt.feedforward(x);
    return 0;
}
