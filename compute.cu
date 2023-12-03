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

	//thread indices
	int row = threadIdx.y;
    int col = threadIdx.x;

	//block indices
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
    
	//Max 12.3.23 12pm
	//declares shared memory arrays to store data to be shared among threads within the block
    __shared__ double sharedMass[BLOCK_SIZE];
    __shared__ vector3 sharedPos[2 * BLOCK_SIZE];
    __shared__ vector3 sharedVel[BLOCK_SIZE];
    __shared__ vector3 sharedAccels[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ vector3 accelSum[BLOCK_SIZE];

	//Max 12.3.23 12pm
	//these arrays load each thread's mass, position, and velocity data into the shared memory
	sharedMass[row] = d_mass[blockRow + row];
    sharedPos[row] = d_hPos[blockRow + row];
    sharedVel[row] = d_hVel[blockRow + row];

	//Max 12.3.23 12pm
	//syncs the threads to ensure each block finished loading data into shared memory before procceeding with computation
	__syncthreads();

	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	for (i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];
	//first compute the pairwise accelerations.  Effect is on the first argument.
	for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}

	__syncthreads();

	//gets the sum of the rows of the matrix
    FILL_VECTOR(accelSum[row], 0, 0, 0);

	//computes pairwise accelerations
    for (int j = 0; j < NUMENTITIES; j++) {
        for (int k = 0; k < 3; k++)
            accelSum[row][k] += accels[row][j][k];
    }

	//
	 for (int k = 0; k < 3; k++) {
        sharedVel[row][k] += accelSum[row][k] * INTERVAL;
        sharedPos[row][k] += sharedVel[row][k] * INTERVAL;
    }

	__syncthreads();

	//copies data back to global
    d_hVel[blockRow + row] = sharedVel[row];
    d_hPos[blockRow + row] = sharedPos[row];



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

