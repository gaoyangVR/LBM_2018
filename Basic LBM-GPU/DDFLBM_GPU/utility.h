#ifndef UTILITY_H
#define UTILITY_H
#include<vector_types.h>

// #define NX 24
// #define NY 24
// #define NZ 96
#define MAXITER 200
#define FLIP_ALPHA 0.95f
#define M_PI       3.14159265358979323846
const float DEGtoRAD = 3.1415926f / 180.0f;

#define TYPEFLUID 0
#define TYPEAIR 1
#define TYPEBOUNDARY 2
#define TYPEVACUUM 3
#define TYPEAIRSOLO 4
#define TYPESOLID 5
#define TYPECNT 6

typedef unsigned int uint;
#define CELL_UNDEF 0xffffffff
#define  NTHREADS 32
#define UNDEF_TEMPERATURE -10000.0f

// struct LBMConstant{
// 	int gnum;
// 	int Re;
// 	float RH0;
// 	float U;
// 	float niu;
// 	float tau;
// 	int Q;
// 	int b;
// 
// 	float3 gmin, gmax, cellsize;
// 
// };

//创建一个方便转换1维与3维数组的数据结构
struct farray{
	float* data;
	int xn, yn, zn;
	farray();
	void setdim(int _xn, int _yn, int _zn){ xn = _xn, yn = _yn, zn = _zn; }

	__host__ __device__ inline float &operator ()(int i, int j, int k)
	{
		return data[i*yn*zn + j*zn + k];
	}
	__host__ __device__ inline float &operator ()(int i)
	{
		return data[i];
	}
	__host__ __device__ inline float &operator [](int i)
	{
		return data[i];
	}
};

//创建一个方便转换1维与3维数组的数据结构
struct charray{
	char* data;
	int xn, yn, zn;
	charray();//{ data = NULL; /*xn=NX; yn=NY; zn=NZ;*/}
	void setdim(int _xn, int _yn, int _zn){ xn = _xn, yn = _yn, zn = _zn; }

	__host__ __device__ inline char &operator ()(int i, int j, int k)
	{
		return data[i*yn*zn + j*zn + k];
	}
	__host__ __device__ inline char &operator ()(int i)
	{
		return data[i];
	}
	__host__ __device__ inline char &operator [](int i)
	{
		return data[i];
	}
};

__host__ __device__ inline void getijk(int &i, int &j, int &k, int &idx, int w, int h, int d)
{
	i = idx / d / h;
	j = idx / d%h;
	k = idx%d;
}
__host__ __device__ inline void getijkq(int &i, int &j, int &k,int &q, int &idx, int w, int h, int d ,int l)
{
	i = idx / d / h/l;
	j = idx / d/l%h;
	k = idx/l%d;
	q = idx%l;

}

enum ERENDERMODE{
	RENDER_HEAT3D = 0,
	RENDER_HEATPLANE,
	RENDER_VELOCITY,
	RENDER_DENSITY,
	RENDER_CNT
};

enum SIMULATIONMODE{
	SIMULATION_WATER = 0,
	SIMULATION_SOLIDCOUPLING,
	SIMULATION_SMOKE,
	SIMULATION_BUBBLE,
	SIMULATION_HEATONLY,
	SIMULATION_CNT
};

enum SCENE{
	SCENE_FLUIDSPHERE = 0,
	SCENE_SMOKE,
	SCENE_BOILING,
	SCENE_MULTIBUBBLE,
	SCENE_DAMBREAK,
	SCENE_MELTING,
	SCENE_MELTINGPOUR,		//melting simulation by pouring water.
	SCENE_FREEZING,
	SCENE_INTERACTION,			//interact with small bubbles, i.e., sub-grid bubbles.
	SCENE_MELTANDBOIL,		//interact with big bubble
	SCENE_HEATTRANSFER,
	SCENE_CNT
};

enum VELOCITYMODEL{
	FLIP = 0,
	CIP,
	HYBRID,
	VELOCITYMODEL_CNT
};

enum ECOLORMODE{
	COLOR_PRESS = 0,
	COLOR_UX,
	COLOR_UY,
	COLOR_UZ,
	COLOR_DIV,	//4
	COLOR_PHI,
	COLOR_MARK,	//6
	COLOR_LS,	//7
	COLOR_TP,	//8
	COLOR_CNT
};

enum TIMESTAT
{
	TIME_DYNAMICS,
	TIME_TRANSITION,
	TIME_DISPLAY,
	TIME_TOTAL,
	TIME_COUNT
};

typedef struct AABB
{
	float xMin, xMax;
	float yMin, yMax;
	float zMin, zMax;
} *pAabb;

//0~ total blue, >=6~total red.
//__host__ __device__ inline float3 mapColorBlue2Red(float v);

struct matrix4
{
	float m[16];
};
struct matrix3x3
{
	float x00, x01, x02;
	float x10, x11, x12;
	float x20, x21, x22;
};

#endif
