#include<helper_cuda.h>
#include <helper_functions.h>

#include <stdio.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

#include "lbm.h"
#include "lbm.cuh"

using namespace std;

const int threadnum = 512;

const int mem_size = (NX) * (NY) * (NZ) * Qm * sizeof(float);

const int blocknum = (int)ceil(((float)(NX)*(NY)*(NZ)) *Qm/ threadnum);

__host__ __device__ inline void getijkq(int &i, int &j, int &k, int &qm, int &idx)
{
	i = idx / (NZ*NY*Qm);
	j = idx / (NZ*Qm) % NY;
	k = idx / Qm%NZ;
	qm = idx%Qm;
}//////////////////////////////////////////////001


__host__ __device__ inline int getidxq(int i, int j, int k, int a)
{
	return (i*NZ*NY*Qm + j*NZ*Qm + k*Qm + a);
}


__host__ __device__ inline void getijk(int &i, int &j, int &k, int &idx)
{
	i = idx / (NZ*NY);
	j = idx / NZ % NY;
	k = idx % NZ;
}


__host__ __device__ inline int getidx(int i, int j, int k)
{
	return (i*NZ*NY + j*NZ + k);
}


__host__ __device__ inline float feq(int i, int j, int k, int a, float R, float T0, float p0, float cv, float *ux, float *uy, float *uz, float *rho, float *T, float *E)
{
	float feq, RT0, eu, u2;
	RT0 = R*T0;
	int e[Qm][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 }, { 1, 1, 0 }, { -1, -1, 0 }, { 1, -1, 0 }, { -1, 1, 0 }, { 1, 0, 1 }, { -1, 0, -1 }, { 1, 0, -1 }, { -1, 0, 1 }, { 0, 1, 1 }, { 0, -1, -1 }, { 0, 1, -1 }, { 0, -1, 1 } };
	float w[Qm] = { 1 / 3.0, 1 / 18.0, 1 / 18.0, 1 / 18.0, 1 / 18.0, 1 / 18.0, 1 / 18.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0 };

	u2 = ux[getidx(i, j, k)] * ux[getidx(i, j, k)] + uy[getidx(i, j, k)] * uy[getidx(i, j, k)] + uz[getidx(i, j, k)] * uz[getidx(i, j, k)];
	eu = e[a][0] * ux[getidx(i, j, k)] + e[a][1] * uy[getidx(i, j, k)] + e[a][2] * uz[getidx(i, j, k)];
	feq = w[a] * rho[getidx(i, j, k)] * (1.0 + eu / RT0 + 0.5*eu*eu / RT0 / RT0 - u2 / 2 / RT0); //总能分布
	return feq;


}
__host__ __device__ inline float heq(int i, int j, int k, int a, float R, float T0, float p0, float cv, float *ux, float *uy, float *uz, float *rho, float *T, float *E)
{
	float  RT0, eu, e2, u2, heq,feq;
	RT0 = R*T0;
	int e[Qm][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 }, { 1, 1, 0 }, { -1, -1, 0 }, { 1, -1, 0 }, { -1, 1, 0 }, { 1, 0, 1 }, { -1, 0, -1 }, { 1, 0, -1 }, { -1, 0, 1 }, { 0, 1, 01 }, { 0, -1, -1 }, { 0, 1, -1 }, { 0, -1, 1 } };
	float w[Qm] = { 1 / 3.0, 1 / 18.0, 1 / 18.0, 1 / 18.0, 1 / 18.0, 1 / 18.0, 1 / 18.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0, 1 / 36.0 };
	e2 = e[a][0] * e[a][0] + e[a][1] * e[a][1] + e[a][2] * e[a][2];
	u2 = ux[getidx(i, j, k)] * ux[getidx(i, j, k)] + uy[getidx(i, j, k)] * uy[getidx(i, j, k)] + uz[getidx(i, j, k)] * uz[getidx(i, j, k)];
	eu = e[a][0] * ux[getidx(i, j, k)] + e[a][1] * uy[getidx(i, j, k)] + e[a][2] * uz[getidx(i, j, k)];
	E[getidx(i, j, k)] = cv*T[getidx(i, j, k)] + u2 / 2.0;
	feq = w[a] * rho[getidx(i, j, k)] * (1.0 + eu / RT0 + 0.5*eu*eu / RT0 / RT0 - u2 / 2 / RT0); //总能分布
	heq = w[a] * p0 * (eu / RT0 + eu*eu / RT0 / RT0 - u2 / RT0 + 0.5*(e2 / RT0 - 3.0)) + E[getidx(i, j, k)] * feq;//总能形式
	return heq;
}

void LBM::initmemset()
{
	copyConstantstoGPU(NX, NY, NZ, Qm, gnum);
	//malloc memory on host
	h_f = (float *)malloc(mem_size);
	h_h = (float *)malloc(mem_size);
	
	h_E = (float *)malloc(mem_size/Qm);
	h_rho = (float *)malloc(mem_size/Qm);
	h_T = (float *)malloc(mem_size/Qm);

	h_vx = (float *)malloc(mem_size/Qm);
	h_vy = (float *)malloc(mem_size/Qm);
	h_vz = (float *)malloc(mem_size/Qm);

	h_vxold = (float *)malloc(mem_size/Qm);
	h_vyold = (float *)malloc(mem_size/Qm);
	h_vzold = (float *)malloc(mem_size/Qm);

	//malloc memory on device

	checkCudaErrors(cudaMalloc((void **)&d_f, mem_size));			//device q & h
	checkCudaErrors(cudaMalloc((void **)&d_h, mem_size));

	//checkCudaErrors(cudaMalloc((void **)&d_tmpf, mem_size));
	//checkCudaErrors(cudaMalloc((void **)&d_tmph, mem_size));

	checkCudaErrors(cudaMalloc((void **)&d_F, mem_size));
	checkCudaErrors(cudaMalloc((void **)&d_H, mem_size));

	cudaMalloc((void **)&d_vx, mem_size / Qm);
	cudaMalloc((void **)&d_vy, mem_size / Qm);
	cudaMalloc((void **)&d_vz, mem_size / Qm);
	cudaMalloc((void **)&d_vxold, mem_size / Qm);
	cudaMalloc((void **)&d_vyold, mem_size / Qm);
	cudaMalloc((void **)&d_vzold, mem_size / Qm);


	cudaMalloc((void **)&d_E, mem_size / Qm);
	cudaMalloc((void **)&d_T, mem_size / Qm);
	cudaMalloc((void **)&d_rho, mem_size / Qm);

	

	printf("init LBM initial memory complete!\n");
	//dim3 grid = dim3(NX, NY);//thread blocks nx/dim  * ny
	//dim3 block = dim3(NX,1 );		//threads per block 


}

void LBM::initsim()
{
	int i, j, k, a;
	for (i = 0; i < NX;i++)
	for (j = 0; j < NY;j++)
	for (k = 0; k < NZ; k++)
	{
		h_vx[getidx(i, j, k)] = 0.0;
		h_vy[getidx(i, j, k)] = 0.0;
		h_vz[getidx(i, j, k)] = 0.0;
		h_rho[getidx(i, j, k)] = RHO;
		h_T[getidx(i, j, k)] = T0;
		for (a = 0; a < Qm;a++)
		{
			h_f[getidxq(i, j, k, a)] = feq(i, j, k, a, R, T0, p0, cv, h_vx, h_vy, h_vz, h_rho, h_T, h_E);
			h_h[getidxq(i, j, k, a)] = heq(i, j, k, a, R, T0, p0, cv, h_vx, h_vy, h_vz, h_rho, h_T, h_E);

		}
	}

	cudaMemcpy(d_f, h_f, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_h, h_h, mem_size, cudaMemcpyHostToDevice);

	cudaMemcpy(d_F, h_f, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_H, h_h, mem_size, cudaMemcpyHostToDevice);

	cudaMemcpy(d_E, h_E, mem_size/Qm, cudaMemcpyHostToDevice);
	cudaMemcpy(d_rho, h_rho, mem_size/Qm, cudaMemcpyHostToDevice);
	cudaMemcpy(d_T, h_T, mem_size/Qm, cudaMemcpyHostToDevice);

	cudaMemcpy(d_vx, h_vx, mem_size/Qm, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vy, h_vy, mem_size/Qm, cudaMemcpyHostToDevice);
	cudaMemcpy(d_vz, h_vz, mem_size/Qm, cudaMemcpyHostToDevice);
	
//	cudaMemcpy(d_vxold, h_vxold, mem_size, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_vyold, h_vyold, mem_size, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_vzold, h_vzold, mem_size, cudaMemcpyHostToDevice);
	//printf("init LBM initial condition complete!");
	//outputuxyz();
	//getchar();
	
}

void LBM::evolution()
{
	evolution_k << <blocknum, threadnum >> >(d_T, d_vx, d_vy, d_vz, d_E, d_rho, d_h, d_f, d_F, d_H, R, cv, wf, wh, T0, p0, U);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
	
	setzero_k << <blocknum, threadnum >> >(d_T, d_vx, d_vy, d_vz, d_E, d_rho, d_h, d_f, d_F, d_H, cv);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	//cudaMemcpy(d_f, d_F, mem_size, cudaMemcpyDeviceToDevice);
//	cudaMemcpy(d_h, d_H, mem_size, cudaMemcpyDeviceToDevice);

	preserve_k << <blocknum, threadnum >> >(d_T, d_vx, d_vy, d_vz, d_E, d_rho, d_h, d_f , d_F, d_H,cv);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	cptmacro_k << <blocknum, threadnum >> >(d_T, d_vx, d_vy, d_vz, d_E, d_rho, d_h, d_f, d_F, d_H, R, T0, p0, cv, U);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");
	
	boundary_k << <blocknum, threadnum >> >(d_T, d_vx, d_vy, d_vz, d_E, d_rho, d_h, d_f, d_F, d_H, R, T0, p0, cv,U);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	initinput_k << <blocknum, threadnum >> >(d_T, d_vx, d_vy, d_vz, d_E, d_rho, cv, U, T_heat);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed");

	cudaMemcpy(h_vx, d_vx, mem_size/Qm, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vy, d_vy, mem_size/Qm, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vz, d_vz, mem_size/Qm, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vxold, d_vxold, mem_size/Qm, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vyold, d_vyold, mem_size/Qm, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_vyold, d_vyold, mem_size/Qm, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_E, d_E, mem_size/Qm, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_T, d_T, mem_size/Qm, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_rho, d_rho, mem_size/Qm, cudaMemcpyDeviceToHost);
}


__global__ void evolution_k(float *T, float *vx, float *vy, float *vz, float *E, float *rho, float *h, float *f, float *F, float *H, float R, float cv, float wf, float wh, float T0, float p0, float U)
{
	int e[Qm][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 }, { 1, 1, 0 }, { -1, -1, 0 }, { 1, -1, 0 }, { -1, 1, 0 }, { 1, 0, 1 }, { -1, 0, -1 }, { 1, 0, -1 }, { -1, 0, 1 }, { 0, 1, 1 }, { 0, -1, -1 }, { 0, 1, -1 }, { 0, -1, 1 } };

	int i, j, k,a;
	int ip, jp, kp;
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < gnum*Qm)
	{
		getijkq(i, j, k,a, idx);
		if (i*j*k != 0 && i < NX -1 && j < NY - 1 && k < NZ - 1)
		{
				
			
				ip = i - e[a][0];
				jp = j - e[a][1];
				kp = k - e[a][2];
				F[getidxq(i, j, k, a)] = f[getidxq(ip, jp, kp, a)] + wf*(feq(ip, jp, kp, a, R, T0, p0, cv, vx, vy, vz, rho, T, E) - f[getidxq(ip, jp, kp, a)]);
				H[getidxq(i, j, k, a)] = h[getidxq(ip, jp, kp, a)] + wh*(heq(ip, jp, kp, a, R, T0, p0, cv, vx, vy, vz, rho, T, E) - h[getidxq(ip, jp, kp, a)]);

			
		}
		
		

	}

}

__global__ void setzero_k(float *T, float *vx, float *vy, float *vz, float *E, float *rho, float *h, float *f, float *F, float *H, float cv)
{
	int i, j, k,a;
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < gnum*Qm)
	{
		getijkq(i, j, k,a,idx);
		if (i*j*k != 0 && i < NX - 1 && j < NY - 1 && k < NZ - 1)
		{
			//?????????????????????????????????????????????????
			rho[getidx(i, j, k)] = 0.0;
			vx[getidx(i, j, k)] = 0.0;
			vy[getidx(i, j, k)] = 0.0;
			vz[getidx(i, j, k)] = 0.0;
			E[getidx(i, j, k)] = 0.0;

			f[getidxq(i, j, k, a)] = F[getidxq(i, j, k, a)];
			h[getidxq(i, j, k, a)] = H[getidxq(i, j, k, a)];
		}
	}
}

__global__ void preserve_k(float *T, float *vx, float *vy, float *vz, float *E, float *rho, float *h, float *f, float *F, float *H, float cv)
{
	int e[Qm][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 }, { 1, 1, 0 }, { -1, -1, 0 }, { 1, -1, 0 }, { -1, 1, 0 }, { 1, 0, 1 }, { -1, 0, -1 }, { 1, 0, -1 }, { -1, 0, 1 }, { 0, 1, 01 }, { 0, -1, -1 }, { 0, 1, -1 }, { 0, -1, 1 } };
	int i, j, k, a;
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < gnum)
	{
	getijk(i, j, k, idx);
	if (i*j*k != 0 && i < NX - 1 && j < NY - 1 && k < NZ - 1)
	{
		//?????????????????????????????????????????????????
	for (int num = 0; num < Qm; num++)
		{
		

			rho[getidx(i, j, k)] += f[getidxq(i, j, k, num)];
			vx[getidx(i, j, k)] += e[num][0] * f[getidxq(i, j, k, num)];
			vy[getidx(i, j, k)] += e[num][1] * f[getidxq(i, j, k, num)];
			vz[getidx(i, j, k)] += e[num][2] * f[getidxq(i, j, k, num)];
			E[getidx(i, j, k)] += h[getidxq(i, j, k, num)];

		}
		/*
		rho[getidx(i, j, k)] = f[getidxq(i, j, k, 0)] + f[getidxq(i, j, k, 1)] + f[getidxq(i, j, k, 2)] + f[getidxq(i, j, k, 3)] + f[getidxq(i, j, k, 4)] + f[getidxq(i, j, k, 5)] + f[getidxq(i, j, k, 6)]
			+ f[getidxq(i, j, k, 7)] + f[getidxq(i, j, k, 8)] + f[getidxq(i, j, k, 9)] + f[getidxq(i, j, k, 10)] + f[getidxq(i, j, k, 11)] + f[getidxq(i, j, k, 12)] + f[getidxq(i, j, k, 13)] + f[getidxq(i, j, k, 14)]
			+ f[getidxq(i, j, k, 15)] + f[getidxq(i, j, k, 16)] + f[getidxq(i, j, k, 17)] + f[getidxq(i, j, k, 18)];

		vx[getidx(i, j, k)] = e[0][0] * f[getidxq(i, j, k, 0)] + e[1][0] * f[getidxq(i, j, k, 1)] + e[2][0] * f[getidxq(i, j, k, 2)] + e[3][0] * f[getidxq(i, j, k, 3)] + e[4][0] * f[getidxq(i, j, k, 4)]
			+ e[5][0] * f[getidxq(i, j, k, 5)] + e[6][0] * f[getidxq(i, j, k, 6)] + e[7][0] * f[getidxq(i, j, k, 7)] + e[8][0] * f[getidxq(i, j, k, 8)] + e[9][0] * f[getidxq(i, j, k, 9)]
			+ e[10][0] * f[getidxq(i, j, k, 10)] + e[11][0] * f[getidxq(i, j, k, 11)] + e[12][0] * f[getidxq(i, j, k, 12)] + e[13][0] * f[getidxq(i, j, k, 13)] + e[14][0] * f[getidxq(i, j, k, 14)]
			+ e[15][0] * f[getidxq(i, j, k, 15)] + e[16][0] * f[getidxq(i, j, k, 16)] + e[17][0] * f[getidxq(i, j, k, 17)] + e[18][0] * f[getidxq(i, j, k, 18)];

		vy[getidx(i, j, k)] = e[0][1] * f[getidxq(i, j, k, 0)] + e[1][1] * f[getidxq(i, j, k, 1)] + e[2][1] * f[getidxq(i, j, k, 2)] + e[3][1] * f[getidxq(i, j, k, 3)] + e[4][1] * f[getidxq(i, j, k, 4)]
			+ e[5][1] * f[getidxq(i, j, k, 5)] + e[6][1] * f[getidxq(i, j, k, 6)] + e[7][1] * f[getidxq(i, j, k, 7)] + e[8][1] * f[getidxq(i, j, k, 8)] + e[9][1] * f[getidxq(i, j, k, 9)]
			+ e[10][1] * f[getidxq(i, j, k, 10)] + e[11][1] * f[getidxq(i, j, k, 11)] + e[12][1] * f[getidxq(i, j, k, 12)] + e[13][1] * f[getidxq(i, j, k, 13)] + e[14][1] * f[getidxq(i, j, k, 14)]
			+ e[15][1] * f[getidxq(i, j, k, 15)] + e[16][1] * f[getidxq(i, j, k, 16)] + e[17][1] * f[getidxq(i, j, k, 17)] + e[18][1] * f[getidxq(i, j, k, 18)];

		vz[getidx(i, j, k)] = e[0][2] * f[getidxq(i, j, k, 0)] + e[1][2] * f[getidxq(i, j, k, 1)] + e[2][2] * f[getidxq(i, j, k, 2)] + e[3][2] * f[getidxq(i, j, k, 3)] + e[4][2] * f[getidxq(i, j, k, 4)]
			+ e[5][2] * f[getidxq(i, j, k, 5)] + e[6][2] * f[getidxq(i, j, k, 6)] + e[7][2] * f[getidxq(i, j, k, 7)] + e[8][2] * f[getidxq(i, j, k, 8)] + e[9][2] * f[getidxq(i, j, k, 9)]
			+ e[10][2] * f[getidxq(i, j, k, 10)] + e[11][2] * f[getidxq(i, j, k, 11)] + e[12][2] * f[getidxq(i, j, k, 12)] + e[13][2] * f[getidxq(i, j, k, 13)] + e[14][2] * f[getidxq(i, j, k, 14)]
			+ e[15][2] * f[getidxq(i, j, k, 15)] + e[16][2] * f[getidxq(i, j, k, 16)] + e[17][2] * f[getidxq(i, j, k, 17)] + e[18][2] * f[getidxq(i, j, k, 18)];

		E[getidx(i, j, k)] = h[getidxq(i, j, k, 0)] + h[getidxq(i, j, k, 1)] + h[getidxq(i, j, k, 2)] + h[getidxq(i, j, k, 3)] + h[getidxq(i, j, k, 4)] + h[getidxq(i, j, k, 5)] + h[getidxq(i, j, k, 6)]
			+ h[getidxq(i, j, k, 7)] + h[getidxq(i, j, k, 8)] + h[getidxq(i, j, k, 9)] + h[getidxq(i, j, k, 10)] + h[getidxq(i, j, k, 11)] + h[getidxq(i, j, k, 12)] + h[getidxq(i, j, k, 13)] + h[getidxq(i, j, k, 14)]
			+ h[getidxq(i, j, k, 15)] + h[getidxq(i, j, k, 16)] + h[getidxq(i, j, k, 17)] + h[getidxq(i, j, k, 18)];

		*/
		
	}
	}
	
}
   
__global__ void cptmacro_k(float *T, float *vx, float *vy, float *vz, float *E, float *rho, float *h, float *f, float *F, float *H, float R, float T0, float p0, float cv, float U)
{
	int i, j, k,a;
//	int e[Qm][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 }, { 1, 1, 0 }, { -1, -1, 0 }, { 1, -1, 0 }, { -1, 1, 0 }, { 1, 0, 1 }, { -1, 0, -1 }, { 1, 0, -1 }, { -1, 0, 1 }, { 0, 1, 01 }, { 0, -1, -1 }, { 0, 1, -1 }, { 0, -1, 1 } };

	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < gnum)
	{
		getijk(i, j, k,idx);
		if (i*j*k != 0 && i < NX - 1 && j < NY - 1 && k < NZ - 1)
		{
			E[getidx(i, j, k)] /= rho[getidx(i, j, k)];
			vx[getidx(i, j, k)] /= rho[getidx(i, j, k)];
			vy[getidx(i, j, k)] /= rho[getidx(i, j, k)];  
			vz[getidx(i, j, k)] /= rho[getidx(i, j, k)];
			

			T[getidx(i, j, k)] = (E[getidx(i, j, k)] - (vx[getidx(i, j, k)] * vx[getidx(i, j, k)] + vy[getidx(i, j, k)] * vy[getidx(i, j, k)]
				+ vz[getidx(i, j, k)] * vz[getidx(i, j, k)]) / 2.0) / cv;
		
		}
	}
}


__global__ void boundary_k(float *T, float *vx, float *vy, float *vz, float *E, float *rho, float *h, float *f, float *F, float *H,float R,float T0,float p0,float cv,float U)
{
	//int e[Qm][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 }, { 1, 1, 0 }, { -1, -1, 0 }, { 1, -1, 0 }, { -1, 1, 0 }, { 1, 0, 1 }, { -1, 0, -1 }, { 1, 0, -1 }, { -1, 0, 1 }, { 0, 1, 01 }, { 0, -1, -1 }, { 0, 1, -1 }, { 0, -1, 1 } };
	int i, j, k, a;
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < gnum*Qm)
	{
		getijkq(i, j, k,a, idx);
		
		if (i*j*k != 0 && i < NX - 1 && j < NY - 1 && k < NZ - 1)
		{
			rho[getidx(NX - 1, j, k)] = rho[getidx(NX - 2, j, k)];
			f[getidxq(NX - 1, j, k, a)] = feq(NX - 1, j, k, a, R, T0, p0, cv, vx, vy, vz, rho, T, E) + f[getidxq(NX - 2, j, k, a)] - feq(NX - 2, j, k, a, R, T0, p0, cv, vx, vy, vz, rho, T, E);
			rho[getidx(0, j, k)] = rho[getidx(1, j, k)];
			f[getidxq(0, j, k, a)] = feq(0, j, k, a, R, T0, p0, cv, vx, vy, vz, rho, T, E) + f[getidxq(1, j, k, a)] - feq(1, j, k, a, R, T0, p0, cv, vx, vy, vz, rho, T, E);
			T[getidx(NX - 1, j, k)] = T[getidx(NX - 2, j, k)];
			h[getidxq(NX - 1, j, k, a)] = h[getidxq(NX - 2, j, k, a)];
			T[getidx(0, j, k)] = T[getidx(1, j, k)];
			h[getidxq(0, j, k, a)] = h[getidxq(1, j, k, a)];
			

			{
			rho[getidx(i, 0, k)] = rho[getidx(i, 1, k)];
			f[getidxq(i, 0, k, a)] = feq(i, 0, k, a, R, T0, p0, cv, vx, vy, vz, rho, T, E) + f[getidxq(i, 1, k, a)] - feq(i, 1, k, a, R, T0, p0, cv, vx, vy, vz, rho, T, E);
			rho[getidx(i, NY - 1, k)] = rho[getidx(i, NY - 2, k)];
			f[getidxq(i, NY - 1, k, a)] = feq(i, NY - 1, k, a, R, T0, p0, cv, vx, vy, vz, rho, T, E) + f[getidxq(i, NY - 2, k, a)] - feq(i, NY - 2, k, a, R, T0, p0, cv, vx, vy, vz, rho, T, E);
			T[getidx(i, 0, k)] = T[getidx(i, 1, k)];
			h[getidxq(i, 0, k, a)] = h[getidxq(i, 1, k, a)];
			T[getidx(i, NY - 1, k)] = T[getidx(i, NY - 2, k)];
			h[getidxq(i, NY - 1, k, a)] = h[getidxq(i, NY - 2, k, a)];
		}

			
			rho[getidx(i, j, NZ - 1)] = rho[getidx(i, j, NZ - 2)];
			f[getidxq(i, j, NZ - 1, a)] = feq(i, j, NZ - 1, a, R, T0, p0, cv, vx, vy, vz, rho, T, E) + f[getidxq(i, j, NZ - 2, a)] - feq(i, j, NZ - 2, a, R, T0, p0, cv, vx, vy, vz, rho, T, E);
			rho[getidx(i, j, 0)] = rho[getidx(i, j, 1)];
			f[getidxq(i, j, 0, a)] = feq(i, j, 0, a, R, T0, p0, cv, vx, vy, vz, rho, T, E) + f[getidxq(i, j, 1, a)] - feq(i, j, 1, a, R, T0, p0, cv, vx, vy, vz, rho, T, E);

			T[getidx(i, j, 0)] = T[getidx(i, j, 1)];
			h[getidxq(i, j, 0, a)] = h[getidxq(i, j, 1, a)];
			T[getidx(i, j, NZ - 1)] = T[getidx(i, 0, NZ - 2)];
			h[getidxq(i, j, NZ - 1, a)] = h[getidxq(i, j, NZ - 2, a)];

			}
			
		}
}
	
__global__ void initinput_k(float *T, float *vx, float *vy, float *vz, float *E, float *rho, float cv, float U, float T_heat)
{
	int i, j, k, a;
	int idx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (idx < gnum)
	{
		getijk(i, j, k, idx);
		if (i >= NX*0.25&&i <= NX*0.75&&j >= 0.25*NY&&j <= 0.75*NY && 1)
		{
			T[getidx(i, j, 0)] = T_heat;
			//	vz[getidx(i, j, 0)] = U;
		}
	}
}


