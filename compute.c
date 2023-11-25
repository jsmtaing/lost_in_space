/*
Group Members:
Joshua Taing
Max Mazal
*/

#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

//initAccels: Initializes the acceleration matrix which is NUMENTITIES squared in size
//Parameters: vector3 *accels, vector3 *vals, int numEntities
//Returns: none
//Notes: i'm thinking we call this first in main, then compute; why am i splitting it up? works like that in my brain
__global__ void initAccels(vector3 *accels, vector3 *vals, int numEntities){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numEntities) {
		accels[idx] = &values[idx*numEntities];
    }
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
__global__ void compute(double *d_mass, vector3 *d_hPos, vector3 *d_hVel){
	
	//You probably noticed I removed memory stuff. Think it is better to do it within nbody.c, in the main function.

    //First compute the pairwise accelerations.  Effect is on the first argument.

    //Sum up the rows of our matrix to get effect on each entity, then update velocity and position.
}