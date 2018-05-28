#include <cuda_runtime.h>

//float feq(int k)
__global__ void LBPropKernelA(float *f1, float *f2, float *f3, float *f4,
	float *f5, float *f6, float *f7, float *f8,
	float *tmpf1, float *tmpf2, float *tmpf3, float *tmpf4,
	float *tmpf5, float *tmpf6, float *tmpf7, float *tmpf8);
__global__ void LBPropKernelB(float *f1, float *f2, float *f3, float *f4,
	float *f5, float *f6, float *f7, float *f8,
	float *tmpf1, float *tmpf2, float *tmpf3, float *tmpf4,
	float *tmpf5, float *tmpf6, float *tmpf7, float *tmpf8);

__global__ void LBCollKernel(float tau, float omega1, float omega2, float omega3,
	float *f0, float *f1, float *f2, float *f3, float *f4,
	float *f5, float *f6, float *f7, float *f8, float vx0,
	float *vx, float *vy, float *vxold, float *vyold);

__global__ void initLBM_k(float *T, float *vx, float *vy, float *vz, float *E, float *rho, float *h, float *f, float R, float T0, float p0,float cv);

__global__ void evolution_k(float *T, float *vx, float *vy, float *vz, float *E, float *rho, float *h, float *f,float *F,float *H, float R, float cv, float wf, float wh,float T0,float p0, float U);

__global__ void preserve_k (float *T, float *vx, float *vy, float *vz, float *E, float *rho, float *h, float *f,float *F, float *H,float cv);

__global__ void boundary_k(float *T, float *vx, float *vy, float *vz, float *E, float *rho, float *h, float *f, float *F, float *H, float R, float T0, float p0, float cv,float U);

__global__ void cptmacro_k(float *T, float *vx, float *vy, float *vz, float *E, float *rho, float *h, float *f, float *F, float *H, float R, float T0, float p0, float cv, float U);

__global__ void setzero_k(float *T, float *vx, float *vy, float *vz, float *E, float *rho, float *h, float *f, float *F, float *H, float cv);

__global__ void initinput_k(float *T, float *vx, float *vy, float *vz, float *E, float *rho, float cv, float U, float T_heat);

void copyConstantstoGPU(int nx, int ny, int nz, int qm, int num);