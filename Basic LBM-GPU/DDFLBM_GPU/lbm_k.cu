#include <cuda_runtime.h>
#include<helper_cuda.h>
#include<helper_math.h>
#include "utility.h"

__constant__ int NX;
__constant__ int NY;
__constant__ int NZ;
__constant__ int Qm;
__constant__ int gnum;



void copyConstantstoGPU(int nx, int ny, int nz,int qm,int num)
{
	checkCudaErrors(cudaMemcpyToSymbol(NX, &nx, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(NY, &ny, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(NZ, &nz, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(Qm, &qm, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(gnum, &num, sizeof(int)));
	
	}



