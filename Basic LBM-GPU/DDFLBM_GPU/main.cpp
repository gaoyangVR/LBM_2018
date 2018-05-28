#pragma warning ( disable : 4819 )

#define GLUT_DISABLE_ATEXIT_HACK 

#include<Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "glew.h"
#include <GL/freeglut.h>
#include "camera.h"
#include"LBM.h"
#include "lbm.cuh"
#include "iostream"
 using namespace std;
Camera gcamera;

int winw = 600, winh = 600;

LBM lbm;

void display()
{
	if (lbm.frame < 100)
	{

		lbm.evolution();
		//统计时间
		
		lbm.DrawFluid();
		lbm.frame++;
		printf("%d \n", lbm.frame);
// 		if (lbm.frame >= 100)
// 		{
// 			getchar();
// 			exit(0);
// 		}

	}

	
	//lbm.outputuxyz();
	
}

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

int main(int argc, char **argv)
{
	printf("Begin: ");

	initopengl(argc, argv);
	printf("initopengl complete.\n");
	lbm.initLBM(); 
	float cx = lbm.gridx*0.5;
	float cy = lbm.gridy*-0.5 / 64;
	float cz = lbm.gridz*0.5;
	gcamera.init(cx, cy, cz, 0.0f, 90.0f, 5.0f, 5.0f, 35.0f, winw, winh, 0.1f, 1000.0f);
		
	glutMainLoop();
	
	return 0;
}
