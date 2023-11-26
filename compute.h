#include "vector.h"

__global__ void initAccels(vector3 *accels, vector3 *vals, int numEntities);
__global__ void compute(double *d_mass, vector3 *d_hPos, vector3 *d_hVel);