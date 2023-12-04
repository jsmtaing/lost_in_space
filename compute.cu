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
#include <stdio.h>

vector3 *d_hPos, *d_hVel, *d_accels, *d_accel_sum;
double *d_mass;

//Function to compute the pairwise accelerations. Effect is on the first argument.
__global__ void comp_PA(vector3 *hPos, double *mass, vector3 *accels){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NUMENTITIES && j < NUMENTITIES){
        if (i == j) {
            FILL_VECTOR(accels[i][j], 0, 0, 0);
        }
        else {
            vector3 distance;
            for (int k = 0; k < 3; k++){
                distance[k] = hPos[i][k] - hPos[j][k];
            }
            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            double magnitude = =sqrt(magnitude_sq);
			double accelmag = -1*GRAV_CONSTANT*mass[j] / magnitude_sq;
			FILL_VECTOR(accels[i][j], accelmag*distance[0] / magnitude, accelmag*distance[1] / magnitude, accelmag*distance[2]/magnitude);
        }
    }
}

//Function to sum rows of the matrix, then update velocity/position.

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
//--Use this function to call parallelized functions above
void compute() {
	
}

