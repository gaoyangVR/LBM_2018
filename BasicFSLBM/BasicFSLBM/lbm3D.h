/// \file lbm3D.h
/// \brief D3Q19 model header file.
//
//	Copyright (c) 2014, Christian B. Mendl
//	All rights reserved.
//	http://christian.mendl.net
//
//	This program is free software; you can redistribute it and/or
//	modify it under the terms of the Simplified BSD License
//	http://www.opensource.org/licenses/bsd-license.php
//
//	Reference:
//	  Nils Thuerey, Physically based animation of free surface flows with the lattice Boltzmann method,
//	  PhD thesis, University of Erlangen-Nuremberg (2007)
//_______________________________________________________________________________________________________________________
//


#ifndef LBM3D_H
#define LBM3D_H

#include "lbm_common.h"
#include "vector3D.h"
#include "util.h"


//_______________________________________________________________________________________________________________________
//


#define SIZE_3D_X			8					//!< x-dimension of the 3D field, must be a power of 2
#define SIZE_3D_Y			16					//!< y-dimension of the 3D field, must be a power of 2
#define SIZE_3D_Z			32					//!< z-dimension of the 3D field, must be a power of 2

#define MASK_3D_X			(SIZE_3D_X - 1)		//!< bit mask for fast modulo 'SIZE_3D_X' operation
#define MASK_3D_Y			(SIZE_3D_Y - 1)		//!< bit mask for fast modulo 'SIZE_3D_Y' operation
#define MASK_3D_Z			(SIZE_3D_Z - 1)		//!< bit mask for fast modulo 'SIZE_3D_Z' operation




class LBM
{
public:
	~LBM()
	{
	}

	float gridx, gridy, gridz;
	real *h_T, T0, RHO;
	float *h_vx, *h_vy, *h_vz, *h_vxold, *h_vyold, *h_vzold;
	real *h_f, *h_h;
	real *h_rho;
	int numsteps;
	int frame;
	bool mpause;
	int rendermode;
	vec3_t gravity;
	void rollrendermode();
	void outputuxyz();
	void DrawFluid();
	void renderheat_3D();
	void renderheat_plane();
	void render_vel();
	void render_dens();

	//_______________________________________________________________________________________________________________________
	///
	/// \brief Distribution functions for D3Q19
	///
	real omega;
	typedef struct
	{
		real f[19];		//!< distribution functions

		int type;		//!< cell type


		// quantities derived from distribution functions
		real rho;		//!< density
		vec3_t u;		//!< velocity

		// keep track of fluid mass exchange
		real mass;		//!< mass

		

	}
	dfD3Q19_t;

	typedef struct
	{
		dfD3Q19_t *df;		//!< array of distribution functions
		real omega;			//!< 1/tau, with tau the relaxation rate to equilibrium
	}
	lbm_field3D_t;
	lbm_field3D_t startfield;
	lbm_field3D_t *fieldevolv;
//------------------------------------------------------------------------------------------------------------------------
	
};	

void DFD3Q19_Init(const int type, const real f[19], LBM::dfD3Q19_t *df);

void evolution(const LBM::lbm_field3D_t *restrict_ startfield, const int numsteps, const vec3_t gravity, LBM::lbm_field3D_t *restrict_ fieldevolv);	
void display();
//_______________________________________________________________________________________________________________________
///
/// \brief LBM field in three dimensions
///



void LBM3DField_Allocate(const real omega, LBM::lbm_field3D_t *field);

void LBM3DField_Free(LBM::lbm_field3D_t *field);


//_______________________________________________________________________________________________________________________
//


/// \brief Access field elements using column-major ordering
static inline LBM::dfD3Q19_t *LBM3DField_Get(const LBM::lbm_field3D_t *field, const int x, const int y, const int z)
{
	return &field->df[x + SIZE_3D_X*(y + SIZE_3D_Y*z)];
}

/// \brief Access field elements using column-major ordering and periodic boundary conditions
static inline LBM::dfD3Q19_t *LBM3DField_GetMod(const LBM::lbm_field3D_t *field, const int x, const int y, const int z)
{
	// bit masking also works for negative integers
	return LBM3DField_Get(field, x & MASK_3D_X, y & MASK_3D_Y, z & MASK_3D_Z);
}


//_______________________________________________________________________________________________________________________
//


void LatticeBoltzmann3DEvolution(const LBM::lbm_field3D_t *restrict_ startfield, const int numsteps, const vec3_t gravity, LBM::lbm_field3D_t *restrict_ fieldevolv);
void CalculateLBMStep3D(const LBM::lbm_field3D_t *restrict_ f0, const vec3_t gravity, LBM::lbm_field3D_t *restrict_ f1);


#endif
