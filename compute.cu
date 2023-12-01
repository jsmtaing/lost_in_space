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
		*accels[idx] = &vals[idx*numEntities];
    }
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
__global__ void compute(vector3 *accels, vector3 *accel_sum, vector3 *hVel, vector3 *hPos, double *mass){
	//You probably noticed I removed memory stuff. Think it is better to do it within nbody.c, in the main function.
    //First compute the pairwise accelerations.  Effect is on the first argument.
	int a = blockDim.y * blockIdx.y + threadIdx.y;
	int b = blockDim.x * blockIdx.x + threadIdx.x;

	if (a < NUMENTITIES && b < NUMENTITIES) {
		if (a == b) {
			FILL_VECTOR(*accels[a][b], 0, 0, 0);
		}
		else {
			vector3 distance;
			for (int k = 0; k < 3; k++) {
				distance[k] = hPos[b][k] - hPos[a][k]; //changed from "hPos[a][k] - hPos[a][k]" to current 
			}
			double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
			double magnitude = sqrt(magnitude_sq);
			double accelmag = -1 * GRAV_CONSTANT * mass[b] / magnitude_sq; //changed from mass j to mass b
			FILL_VECTOR(*accels[a][b], accelmag * distance[0] / magnitude, accelmag * distance[1] / magnitude, accelmag * distance[2] / magnitude);
		}
	}

	//synchronizes the threads before procceeding
	__syncthreads();

    //Sum up the rows of our matrix to get effect on each entity.
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int d = blockIdx.y;

	if (c < NUMENTITIES) {
		FILL_VECTOR(accel_sum[c], 0, 0, 0);
		for (int k = 0; k < NUMENTITIES; k++) {
			accel_sum[c][d] += *accels[c][k][d];
		}
		//Then update velocity and position.
		hVel[c][d] += accel_sum[c][d];
		hPos[c][d] += hVel[c][d] * INTERVAL;
	}
}

