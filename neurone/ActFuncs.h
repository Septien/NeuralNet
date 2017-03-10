#ifndef ACTFUNCS_H_INCLUDED
#define ACTFUNCS_H_INCLUDED

#include <math.h>

double stepAF(double y)
{
    return (y < 0) ? 0 : 1;
}

double sigmoidAF(double y)
{
    return 1.0 / ( 1.0 + exp(-0.5 * y));
}

double piecewiceAF(double y)
{
    return (y >= 0.5 ? 1 : (y <= 0.5 ? 0 : y));
}

double signumAF(double y)
{
    double e = 0.00001;     //Epsilon
    return ((y - e) > 0 ? 1 : ( (y + e) < 0 ? -1 : 0))
}

double tanhAF(double y)
{
    return tanh(y);
}

#endif // ACTFUNCS_H_INCLUDED
