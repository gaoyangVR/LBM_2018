#ifndef LBM_H
#define LBM_H

#endif

#define GETIDX(i,j,k) (i*NZ*NY + j*NZ + k)


const int NX = 32;
const int NY = 32;
const int NZ = 32;
const int Qm = 19;

const int gnum = NX*NY*NZ;



class LBM
{
public:
	~LBM()
	{
	}

public:
	float *d_f, *d_h;
	float *d_tmpf, *d_tmph;
	float *h_f, *h_h;
	float *d_F, *d_H;

	float *d_vx, *d_vy, *d_vz, *d_vxold, *d_vyold , *d_vzold;
	float *h_vx, *h_vy, *h_vz, *h_vxold, *h_vyold, *h_vzold;

	float *h_rho,*d_rho, *h_E,*d_E, *h_T,*d_T;

public:
	int Re;
	float RHO;	//density
	float U;
	float niu;
	float tau_f, tau_h;	//无量纲松弛时间
	int  frame;//控制初始速度

	float T0;
	float R;
	float p0;
	float cv;
	float Pr;
	float Ra;
	float delta_T;
	float T_heat;
	float total_E;
	float wf, wh;

public:
	float omega1, omega2, omega3;
	float e[Qm][3];

	float gridx, gridy, gridz;


	bool mpause;


	void evolution();
	void outputuxyz();
	void DrawFluid();
	float T_to_E(int i, int j, int k);
	float u2();
	void renderheat_3D();
	void renderheat_plane();
	void render_vel();
	void render_dens();
	void rollrendermode();

	void solidmotion();
	int rendermode;

	void initmemset();
	void initparam();
	void initsim();
	void initLBM();
	

};