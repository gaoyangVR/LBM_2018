#include"lbm.h"
#include"lbm.cuh"

#include <stdio.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
using namespace std;

void LBM::initparam()
{

	//factors used in equilibrium functions
	delta_T = 0.1;
	Pr = 0.7;
	Ra = 10000;
	RHO = 0.2;
	U = 0.20;
	R = 8.3144;	//气体普适常量
	T0 = 0.04;
	p0 = RHO * R *T0;
	tau_f = 0.50;	//????存疑
	tau_h = tau_f / Pr;
	niu = tau_f * p0;
	cv = (3 + 3)*R / 2.0;
	frame = 0;
	wf = 2 * delta_T / (2 * tau_f + delta_T);
	wh = 2 * delta_T / (2 * tau_h + delta_T);
	total_E = 0.0;
	T_heat = 200;
	



	cout << " Init parameter complete." << endl;
	
}

void LBM::initLBM()
{
	initmemset();
	initparam();
	initsim();
	printf("init simulation complete!");
	getchar();

}