//	Implementation of the lattice Boltzmann method (LBM) using the D2Q9 and D3Q19 models
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

// Demonstration file in 3D with free surfaces (fluid to gas interface)

#include"camera.h"
#include "lbm3D.h"
#include"glew.h"
#include <GL/freeglut.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
//#include "lbm3D.cpp"
#if defined(_WIN32) & (defined(DEBUG) | defined(_DEBUG))
#include <crtdbg.h>
#endif
using namespace std;
extern Camera gcamera;
LBM lbm;
int winw = 600, winh = 600;

//keyboard control
void keyboard_func(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:

		break;
	case ' ':
		lbm.mpause = !lbm.mpause;
		break;
	case 'R': case 'r':
		lbm.rollrendermode();
		break;
		// 	case 'C': case 'c':
		// 		gfluid.rollcolormode(1);
		// 		break;
		// 	case 'V': case 'v':
		// 		gfluid.rollcolormode(-1);
		// 		break;
		// 	case 'T': case 't':
		// 		gfluid.m_btimer = !gfluid.m_btimer;
		// 		break;
		// 	case 'M': case 'm':
		// 		gcamera.resetCamto();
		// 		break;
		// 	case 'p': case'P':
		// 	{
		// 				  if (gfluid.renderpartiletype == TYPEAIR)
		// 					  gfluid.renderpartiletype = TYPEFLUID;
		// 				  else if (gfluid.renderpartiletype == TYPEFLUID)
		// 					  gfluid.renderpartiletype = TYPEAIR;
		// 				  break;
		// 	}
		// 	case 'O': case 'o':
		// 		gfluid.boutputpovray = !gfluid.boutputpovray;
		// 		break;
		// 	case 'I': case 'i':
		// 		gfluid.mRecordImage = !gfluid.mRecordImage;
		// 		break;
		// 	case 'S': case 's':
		// 		gfluid.m_bSmoothMC = !gfluid.m_bSmoothMC;
		// 		break;
		// 	case 'D': case 'd':
		// 		gfluid.m_DistanceFuncMC = (gfluid.m_DistanceFuncMC + 1) % 2;
		// 		break;
	}
}
//...................................................................................................

//------------------------------------------------------------------------
//mouse control

void mouse_click(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
		gcamera.mode = CAMERA_MOVE;
	else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
		gcamera.mode = CAMERA_MOVECENTER;
	else
		gcamera.mode = CAMERA_NONE;

	gcamera.last_x = x;
	gcamera.last_y = y;
}

void mouse_move(int x, int y)
{
	gcamera.mousemove(x, y);
}

void mousewheel(int button, int dir, int x, int y)
{
	gcamera.mousewheel(dir*0.1f);
}
//_________________________________________________________________________

//_________________________________________________
//OpenGL init



void initopengl(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowPosition(620, 0);
	glutInitWindowSize(winw, winh);
	glutCreateWindow("fluidSimulation");
	GLenum err = glewInit();
	if (err != GLEW_OK)
		printf("\nGLEW is not available!!!\n\n");

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard_func);
	glutMouseFunc(mouse_click);
	glutMotionFunc(mouse_move);
	glutMouseWheelFunc(mousewheel);
}

//_________________________________________________

//_______________________________________________________________________________________________________________________
///
/// \brief Calculate the average mass of a field
///
static inline real LBM3DField_AverageMass(const LBM::lbm_field3D_t *field)
{
	int x, y, z;

	real mass = 0;
	for (z = 0; z < SIZE_3D_Z; z++)
	{
		for (y = 0; y < SIZE_3D_Y; y++)
		{
			for (x = 0; x < SIZE_3D_X; x++)
			{
				// current cell
				const LBM::dfD3Q19_t *df = LBM3DField_Get(field, x, y, z);
				mass += df->mass;
			}
		}
	}

	// normalization
	mass /= (SIZE_3D_X*SIZE_3D_Y*SIZE_3D_Z);

	return mass;
}


//_______________________________________________________________________________________________________________________
///
/// \brief Calculate the average velocity of a field
///
static inline vec3_t LBM3DField_AverageVelocity(const LBM::lbm_field3D_t *field)
{
	int x, y, z;

	real rho = 0;
	vec3_t vel = { 0 };
	for (z = 0; z < SIZE_3D_Z; z++)
	{
		for (y = 0; y < SIZE_3D_Y; y++)
		{
			for (x = 0; x < SIZE_3D_X; x++)
			{
				// current cell
				const LBM::dfD3Q19_t *df = LBM3DField_Get(field, x, y, z);

				rho += df->rho;

				// weighted contribution to overall velocity
				vel.x += df->rho * df->u.x;
				vel.y += df->rho * df->u.y;
				vel.z += df->rho * df->u.z;
			}
		}
	}

	// normalization
	real s = 1 / rho;
	vel.x *= s;
	vel.y *= s;
	vel.z *= s;

	return vel;
}


//_______________________________________________________________________________________________________________________
//


int main(int argc, char ** argv)
{
	int i;
	int x, y, z;
	lbm.omega = 0.2f;
	lbm.numsteps = 1280;
	const vec3_t gravity = { 0, 0, (real)(-0.1) };

	initopengl(argc, argv);
	printf("initopengl complete.\n");
	float cx = lbm.gridx*0.5;
	float cy = lbm.gridy*-0.5 / 64;
	float cz = lbm.gridz*0.5;
	gcamera.init(cx, cy, cz, 0.0f, 90.0f, 5.0f, 5.0f, 35.0f, winw, winh, 0.1f, 1000.0f);

	printf("dimensions: %d x %d x %d\n", SIZE_3D_X, SIZE_3D_Y, SIZE_3D_Z);
	printf("omega:      %g\n", lbm.omega);
	printf("numsteps:   %d\n", lbm.numsteps);
	printf("gravity:    (%g, %g, %g)\n", gravity.x, gravity.y, gravity.z);
	printf("_______________________________________________________________\n");
	// enable run-time memory check for debug builds
#if defined(_WIN32) & (defined(DEBUG) | defined(_DEBUG))
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	// load initial cell types from disk
	int t0[SIZE_3D_X*SIZE_3D_Y*SIZE_3D_Z];
	int hr = ReadData("../test/lbm_sim3D_freesurf_t0.dat", t0, sizeof(int), SIZE_3D_X*SIZE_3D_Y*SIZE_3D_Z);
	if (hr < 0) {
		fprintf(stderr, "File containing initial cell types not found, exiting...\n");
		return hr;
	}

	// load initial distribution function data from disk
	real f0[SIZE_3D_X*SIZE_3D_Y*SIZE_3D_Z * 19];
	hr = ReadData("../test/lbm_sim3D_freesurf_f0.dat", f0, sizeof(real), SIZE_3D_X*SIZE_3D_Y*SIZE_3D_Z * 19);
	if (hr < 0) {
		fprintf(stderr, "File containing distribution function data not found, exiting...\n");
		return hr;
	}

	// allocate and initialize start field
	
	LBM3DField_Allocate(lbm.omega, &lbm.startfield);
	for (z = 0; z < SIZE_3D_Z; z++)
	{
		for (y = 0; y < SIZE_3D_Y; y++)
		{
			for (x = 0; x < SIZE_3D_X; x++)
			{
				// data stored with row-major ordering
				i = (x*SIZE_3D_Y + y)*SIZE_3D_Z + z;
				DFD3Q19_Init(t0[i], &f0[19 * i], LBM3DField_Get(&lbm.startfield, x, y, z));
			}
		}
	}

	//LBM::lbm_field3D_t *fieldevolv;
	lbm.fieldevolv = (LBM::lbm_field3D_t *)malloc(lbm.numsteps*sizeof(LBM::lbm_field3D_t));

// prepare main calculation
	LatticeBoltzmann3DEvolution(&lbm.startfield, lbm.numsteps, gravity, lbm.fieldevolv);
	
	// start timer
	clock_t t_start = clock();

	glutMainLoop();

	clock_t t_end = clock();
	double cpu_time = (double)(t_end - t_start) / CLOCKS_PER_SEC;
	printf("finished simulation, CPU time: %g\n\n", cpu_time);

	// check mass conservation
	real mass_init = LBM3DField_AverageMass(&lbm.fieldevolv[0]);
	real mass_final = LBM3DField_AverageMass(&lbm.fieldevolv[lbm.numsteps - 1]);
	printf("initial average mass: %g\n", mass_init);
	printf("final   average mass: %g\n", mass_final);
	printf("relative difference:  %g\n", fabsf(mass_final / mass_init - 1));
	printf("(mass not exactly conserved due to excess mass which cannot be distributed)\n\n");

	// check momentum conservation
	vec3_t vel_init = LBM3DField_AverageVelocity(&lbm.fieldevolv[0]);
	vec3_t vel_final = LBM3DField_AverageVelocity(&lbm.fieldevolv[lbm.numsteps - 1]);
	printf("initial average velocity: (%g, %g, %g)\n", vel_init.x, vel_init.y, vel_init.z);
	printf("final   average velocity: (%g, %g, %g)\n", vel_final.x, vel_final.y, vel_final.z);
	vec3_t vel_diff = {
		vel_final.x - vel_init.x,
		vel_final.y - vel_init.y,
		vel_final.z - vel_init.z
	};
	printf("norm of difference: %g\n", Vec3_Norm(vel_diff));
	printf("(velocity not conserved due to clamping to maximum velocity)\n\n");

	// clean up
	for (i = 0; i < lbm.numsteps; i++)
	{
		LBM3DField_Free(&lbm.fieldevolv[i]);
	}
	free(lbm.fieldevolv);
	LBM3DField_Free(&lbm.startfield);

	return 0;
}


void display(/*const LBM::lbm_field3D_t *restrict_ startfield, const int numsteps, const vec3_t gravity, LBM::lbm_field3D_t *restrict_ fieldevolv*/)
{

	evolution(&lbm.startfield, lbm.numsteps, lbm.gravity, lbm.fieldevolv);
	//	lbm.DrawFluid();
	//	lbm.outputuxyz();
	glutLeaveMainLoop();
	//double cpu_time = (double)(t_end - t_start) / CLOCKS_PER_SEC;
	//	printf("finished simulation, CPU time: %g\n\n", cpu_time);

	// check mass conservation
	real mass_init = LBM3DField_AverageMass(&lbm.fieldevolv[0]);
	real mass_final = LBM3DField_AverageMass(&lbm.fieldevolv[lbm.numsteps - 1]);
	printf("initial average mass: %g\n", mass_init);
	printf("final   average mass: %g\n", mass_final);
	printf("relative difference:  %g\n", fabsf(mass_final / mass_init - 1));
	printf("(mass not exactly conserved due to excess mass which cannot be distributed)\n\n");

	// check momentum conservation
	vec3_t vel_init = LBM3DField_AverageVelocity(&lbm.fieldevolv[0]);
	vec3_t vel_final = LBM3DField_AverageVelocity(&lbm.fieldevolv[lbm.numsteps - 1]);
	printf("initial average velocity: (%g, %g, %g)\n", vel_init.x, vel_init.y, vel_init.z);
	printf("final   average velocity: (%g, %g, %g)\n", vel_final.x, vel_final.y, vel_final.z);
	vec3_t vel_diff = {
		vel_final.x - vel_init.x,
		vel_final.y - vel_init.y,
		vel_final.z - vel_init.z
	};
	printf("norm of difference: %g\n", Vec3_Norm(vel_diff));
	printf("(velocity not conserved due to clamping to maximum velocity)\n\n");

	// clean up
	for (int i = 0; i < lbm.numsteps; i++)
	{
		LBM3DField_Free(&lbm.fieldevolv[i]);
	}
	free(lbm.fieldevolv);
	LBM3DField_Free(&lbm.startfield);


}