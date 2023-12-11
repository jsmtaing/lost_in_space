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

//temporary library
#include <stdio.h>

vector3 *d_hPos, *d_hVel, *d_accels;
double *d_mass;

//Function that computes the pairwise accelerations. Effect is on the first argument.
__global__ void comp_PA(vector3 *hPos, double *mass, vector3 *accels){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= NUMENTITIES) {
		return;
	}
    
    //This part was just C+P'd from the original compute.c -- only change is
    //that it's not a for loop, since it should be looping in the for loop in nbody.c's main instead.
    if (i < NUMENTITIES && j < NUMENTITIES){
        if (i == j) {
            FILL_VECTOR(accels[i * NUMENTITIES + j], 0, 0, 0);
        }
        else {
            vector3 distance;
            for (int k = 0; k < 3; k++){
                distance[k] = hPos[i][k] - hPos[j][k];
            }
            double magnitude_sq = distance[0]*distance[0] + distance[1]*distance[1] + distance[2]*distance[2];
            double magnitude = sqrt(magnitude_sq);
			double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
			FILL_VECTOR(accels[i * NUMENTITIES + j], accelmag*distance[0]/magnitude, accelmag*distance[1]/magnitude, accelmag*distance[2]/magnitude);
        }
    }
}

//Function to sum rows of the matrix, then update velocity/position.
__global__ void sum_update(vector3* hVel, vector3* hPos, vector3* accels){
    int i = threadIdx.x;
    if (i >= NUMENTITIES) {
		return;
	}

        vector3 accel_sum = {0, 0, 0};
        for (j = 0; j < NUMENTITIES ; j++){
            for (k = 0; k < 3; k++){
                accel_sum[k] += accels[i * NUMENTITIES + j][k];
            }
        }
        //compute the new velocity based on the acceleration and time interval
        //compute the new position based on the velocity and time interval
        for (k = 0; k < 3; k++){
            hVel[i][k] += accel_sum[k] * INTERVAL;
            hPos[i][k] += hVel[i][k] * INTERVAL;
        }
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
//Use this function to call parallelized functions above
void compute() {
	dim3 blockDim(16, 16);
	dim3 gridDim((NUMENTITIES + blockDim.x - 1) / blockDim.x, (NUMENTITIES + blockDim.y - 1) / blockDim.y);

    comp_PA<<<gridDim, blockDim>>>(d_hPos, d_mass, d_accels);
    cudaDeviceSynchronize();
    //cudaError_t err = cudaGetLastError();
    //if (err != cudaSuccess) 
    //    printf("Error: %s\n", cudaGetErrorString(err));

    sum_update<<<gridDim, blockDim>>>(d_hVel, d_hPos, d_accels);
    //cudaDeviceSynchronize();

    cudaMemcpy(hPos, d_hPos, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, d_hVel, sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
    //cudaMemcpy(mass, d_mass, sizeof(double)*NUMENTITIES, cudaMemcpyDeviceToHost);
}
