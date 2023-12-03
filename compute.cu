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
__global__ void compute(vector3 *d_accels, vector3 *d_accel_sum, vector3 *d_hVel, vector3 *d_hPos, double *d_mass) {
	//thread indices
	int row = threadIdx.y;
    int col = threadIdx.x;

	//block indices
	int blockRow = blockIdx.y;
	// int blockCol = blockIdx.x;
    
	//Max 12.3.23 12pm
	//declares shared memory arrays to store data to be shared among threads within the block
    __shared__ double sharedMass[BLOCK_SIZE];
    __shared__ vector3 sharedPos[BLOCK_SIZE];
    __shared__ vector3 sharedVel[BLOCK_SIZE];
    __shared__ vector3 sharedAccels[BLOCK_SIZE * BLOCK_SIZE];


	//Max 12.3.23 12pm
	//these arrays load each thread's mass, position, and velocity data into the shared memory (respectively)
	sharedMass[row] = d_mass[blockRow + row];
    for (int k = 0; k < 3; k++)
    	sharedPos[row][k] = d_hPos[blockRow + row][k];
    for (int k = 0; k < 3; k++)
    	sharedVel[row][k] = d_hVel[blockRow + row][k];

	//Max 12.3.23 12pm
	//syncs the threads to ensure each block finished loading data into shared memory before procceeding with computation
	__syncthreads();


	//first compute the pairwise accelerations.  Effect is on the first argument.
	vector3 tempAccel = {0, 0, 0};
    for (int j = 0; j < BLOCK_SIZE; j++){
        vector3 distance;
        for (int k = 0; k < 3; k++)
			distance[k] = sharedPos[row][k] - sharedPos[j][k];

        double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
        double magnitude = sqrt(magnitude_sq);
        double accelmag = -1 * GRAV_CONSTANT * sharedMass[j] / magnitude_sq;

        for (int k = 0; k < 3; k++)
            tempAccel[k] += accelmag * distance[k] / magnitude;
   	}

	__syncthreads();

	//gets the sum of the rows of the matrix
	for (int k = 0; k < 3; k++)
    	sharedAccels[row * BLOCK_SIZE + col][k] = tempAccel[k];


	__syncthreads();

	//computes pairwise accelerations
	vector3 accelSum = {0, 0, 0};
    for (int j = 0; j < BLOCK_SIZE; j++) {
        for (int k = 0; k < 3; k++)
            accelSum[k] += sharedAccels[row * BLOCK_SIZE + j][k];
    }

	//
	 for (int k = 0; k < 3; k++) {
        sharedVel[row][k] += accelSum[k] * INTERVAL;
        sharedPos[row][k] += sharedVel[row][k] * INTERVAL;
    }

	__syncthreads();

	//copies data back to global
    for (int k = 0; k < 3; k++) {
		d_hVel[blockRow + row][k] = sharedVel[row][k];
		d_hPos[blockRow + row][k] = sharedPos[row][k];
	}

	vector3* values = (vector3*)malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
    vector3** accels = (vector3**)malloc(sizeof(vector3*) * NUMENTITIES);
    for (int i = 0; i < NUMENTITIES; i++)
        accels[i] = &values[i * NUMENTITIES];


	// //sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	// for (i=0;i<NUMENTITIES;i++){
	// 	vector3 accel_sum={0,0,0};
	// 	for (j=0;j<NUMENTITIES;j++){
	// 		for (k=0;k<3;k++)
	// 			accel_sum[k]+=accels[i][j][k];
	// 	}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		// for (k=0;k<3;k++){
		// 	hVel[i][k]+=accel_sum[k]*INTERVAL;
		// 	hPos[i][k]+=hVel[i][k]*INTERVAL;
		// }

	free(accels);
	free(values);
}

