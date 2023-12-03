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

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
__global__ void compute(double *d_mass, vector3 *d_hPos, vector3 *d_hVel){
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i,j,k;

	//thread / block indices
	//Max 12.3.23 12pm
	//swapped variables row and column (row corresponds to blockIdx.x and vice versa)
	int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
	//Max 12.3.23 12pm
	//declares shared memory arrays to store data to be shared among threads within the block
	__shared__ double sharedMass[BLOCK_SIZE];
	__shared__ vector3 sharedPos[BLOCK_SIZE];
    __shared__ vector3 sharedVel[BLOCK_SIZE];

	//Max 12.3.23 12pm
	//these arrays load each thread's mass, position, and velocity data into the shared memory
	sharedMass[threadIdx.x] = d_mass[col];
    sharedPos[threadIdx.x] = d_hPos[col];
    sharedVel[threadIdx.x] = d_hVel[col];

	//Max 12.3.23 12pm
	//syncs the threads to ensure each block finished loading data into shared memory before procceeding with computation
	__syncthreads();

	__shared__ vector3 accels[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ vector3 accel_sum[BLOCK_SIZE];

	accel_sum[threadIdx.x] = {0, 0, 0};

	__syncthreads();


	//first compute the pairwise accelerations.  Effect is on the first argument.
	for (int j = 0; j < NUMENTITIES; j++) {
        if (row != j) {
            vector3 distance;
            for (int k = 0; k < 3; k++)
                distance[k] = sharedPos[threadIdx.x][k] - d_hPos[j][k];

            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            double magnitude = sqrt(magnitude_sq);
            double accelmag = -1 * GRAV_CONSTANT * sharedMass[threadIdx.x] / magnitude_sq;

            for (int k = 0; k < 3; k++)
				accel_sum[threadIdx.x][k] += accelmag * distance[k] / magnitude;
        }
    }

	__syncthreads();

	
	 for (int k = 0; k < 3; k++) {
        sharedVel[threadIdx.x][k] += accel_sum[k] * INTERVAL;
        sharedPos[threadIdx.x][k] += sharedVel[threadIdx.x][k] * INTERVAL;
    }

    __syncthreads();

    d_hVel[col] = sharedVel[threadIdx.x];
    d_hPos[col] = sharedPos[threadIdx.x];
}

