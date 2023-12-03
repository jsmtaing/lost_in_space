#include "vector.h"

__global__ void initAccels(vector3 *accels, vector3 *vals, int numEntities);
__global__ void compute(vector3 *d_accels, vector3 *d_accel_sum, vector3 *d_hVel, vector3 *d_hPos, double *d_mass);