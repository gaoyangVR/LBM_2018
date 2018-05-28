//屏蔽掉unicode的warning.
#pragma warning ( disable : 4819 )

#include<Windows.h>
#include "camera.h"
#include "inc/glm-0.9.4.0/glm/glm.hpp"
#include "inc/glm-0.9.4.0/glm/gtc/matrix_transform.hpp"
#include "glew.h"
#include <GL/freeglut.h>
#include "lbm3D.h"
#include"iosfile.h"


using namespace glm;

 Camera gcamera;

int getid(int i, int j, int k)
{
	return (i*SIZE_3D_X*SIZE_3D_Y + j*SIZE_3D_Z + k);
}

void setLookAtFromCam()
{
	gluLookAt(gcamera.cam_from.x, gcamera.cam_from.y, gcamera.cam_from.z,
		gcamera.cam_to.x, gcamera.cam_to.y, gcamera.cam_to.z,
		gcamera.cam_up.x, gcamera.cam_up.y, gcamera.cam_up.z);
}

void setProjectionFromCam()
{
	gluPerspective(gcamera.cam_fov, gcamera.cam_aspect, gcamera.nearplane, gcamera.farplane);
}

void setglmLookAtMatrixFromCam(mat4 &mat)
{
	mat = glm::lookAt(vec3(gcamera.cam_from.x, gcamera.cam_from.y, gcamera.cam_from.z),
		vec3(gcamera.cam_to.x, gcamera.cam_to.y, gcamera.cam_to.z),
		vec3(gcamera.cam_up.x, gcamera.cam_up.y, gcamera.cam_up.z));
}

void setglmProjMatrixFromCam(mat4 &mat)
{
	mat = glm::perspective(gcamera.cam_fov, gcamera.cam_aspect, gcamera.nearplane, gcamera.farplane);
}

void render()
{
	//glClearColor( 1.0f, 1.0f, 1.0f, 1.0f );
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//  	glEnable (GL_LINE_SMOOTH );
	// 	glHint (GL_LINE_SMOOTH, GL_NICEST);
	glLineWidth(2);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	setProjectionFromCam();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	setLookAtFromCam();

	glUseProgram(0);
	glDisable(GL_BLEND);
	//glDisable( GL_LIGHTING );
	glColor3f(1.0f, 0.f, 0.f);

	//render the simple cube boundary
	glEnable(GL_DEPTH_TEST);
	glPushMatrix();
	//glTranslatef(hparam.gmax.x*0.5f, hparam.gmax.y*0.5f, hparam.gmax.z*0.5f);
	glScalef(SIZE_3D_X, SIZE_3D_Y, SIZE_3D_Z);
	glDisable(GL_LIGHTING);
	//if (mscene != SCENE_HEATTRANSFER)
	glutWireCube(1.0f);
	glPopMatrix();


	glEnable(GL_PROGRAM_POINT_SIZE);


	//set uniform variables.
	glm::mat4 model = glm::mat4(1.0f), view, proj;
	setglmLookAtMatrixFromCam(view);
	setglmProjMatrixFromCam(proj);
	mat4 MVP = proj*view*model;		//这里的顺序很重要
	mat4 MV = view*model;
	glEnable(GL_DEPTH_TEST);




	glutPostRedisplay();
	glutSwapBuffers();
}

void initlight()
{
	// good old-fashioned fixed function lighting
	float black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	float ambient[] = { 0.1f, 0.1f, 0.1f, 1.0f };
	float diffuse[] = { 0.9f, 0.9f, 0.9f, 1.0f };
	float lightPos[] = { 0.0f, 0.0f, 1.0f, 0.0f };

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);

	glLightfv(GL_LIGHT0, GL_AMBIENT, white);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
	glLightfv(GL_LIGHT0, GL_SPECULAR, white);
	glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);

	glEnable(GL_LIGHT0);
}


void LBM::DrawFluid()
{
	//glClearColor( 1.0f, 1.0f, 1.0f, 1.0f );
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//  	glEnable (GL_LINE_SMOOTH );
	// 	glHint (GL_LINE_SMOOTH, GL_NICEST);
	glLineWidth(2);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	setProjectionFromCam();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	setLookAtFromCam();

	glUseProgram(0);
	glDisable(GL_BLEND);
	//glDisable( GL_LIGHTING );
	glColor3f(1.0f, 0.f, 0.f);
	glTranslatef(gridx*0.5f, gridy*0.5f, gridz*0.5f);
	//render the simple cube boundary
	glEnable(GL_DEPTH_TEST);
	glPushMatrix();
	//	
	glScalef(gridx, gridy, gridz);
	glDisable(GL_LIGHTING);
	//if (mscene != SCENE_HEATTRANSFER)
	glutWireCube(1.0f);
	glPopMatrix();


	glEnable(GL_PROGRAM_POINT_SIZE);

	glTranslatef(-SIZE_3D_X / 2, -SIZE_3D_Y / 2, -SIZE_3D_Z / 2);
	//set uniform variables.
	glm::mat4 model = glm::mat4(1.0f), view, proj;
	setglmLookAtMatrixFromCam(view);
	setglmProjMatrixFromCam(proj);
	mat4 MVP = proj*view*model;		//这里的顺序很重要
	mat4 MV = view*model;
	glEnable(GL_DEPTH_TEST);

 	if (rendermode == RENDER_HEAT3D)render_vel(); //renderheat_3D();
  	if (rendermode == RENDER_HEATPLANE)render_vel(); //renderheat_plane();
  	if (rendermode == RENDER_VELOCITY)render_vel();
  	if (rendermode == RENDER_DENSITY)render_vel(); //render_dens();
	glutPostRedisplay();
	glutSwapBuffers();
}

void LBM::renderheat_3D()
{
	int i, j, k;

	float color[3] = { 0, 0, 0 };
	float tmpcolor;
	for (i = 0; i < SIZE_3D_X; i++)
	for (j = 0; j < SIZE_3D_Y; j++)
	for (k = 0; k < SIZE_3D_Z; k++)
	{
		tmpcolor = (h_T[getid(i, j, k)] - T0) / 10.0;
		//float tmpcolor = (lbm.T[i][j][k]*1.0);
		if (tmpcolor < 0)
			color[2] = 1;

		int ic = (int)tmpcolor;
		float f = tmpcolor - ic;
		switch (ic)
		{
		case 0:
		{
				  color[0] = 0;
				  color[1] = f / 2;
				  color[2] = 1;
		}
			break;
		case 1:
		{

				  color[0] = 0;
				  color[1] = f / 2 + 0.5f;
				  color[2] = 1;
		}
			break;
		case 2:
		{
				  color[0] = f / 2;
				  color[1] = 1;
				  color[2] = 1 - f / 2;
		}
			break;
		case 3:
		{
				  color[0] = f / 2 + 0.5f;
				  color[1] = 1;
				  color[2] = 0.5f - f / 2;
		}
			break;
		case 4:
		{
				  color[0] = 1;
				  color[1] = 1.0f - f / 2;
				  color[2] = 0;
		}
			break;
		case 5:
		{
				  color[0] = 1;
				  color[1] = 0.5f - f / 2;
				  color[2] = 0;
		}
			break;
		default:
		{
				   color[0] = 1;
				   color[1] = 0;
				   color[2] = 0;
		}
			break;
		}

		{
			glColor4f(color[0], color[1], color[2], 0.5);//设置颜色
			glBegin(GL_QUADS);
			glVertex3f((float)i - 0.1, (float)j - 0.1, (float)k);
			glVertex3f((float)i - 0.1, (float)j + 0.1, (float)k);
			glVertex3f((float)i + 0.1, (float)j + 0.1, (float)k);
			glVertex3f((float)i + 0.1, (float)j - 0.1, (float)k);
			glEnd();
			glBegin(GL_QUADS);
			glVertex3f((float)i - 0.1, (float)j, (float)k - 0.1);
			glVertex3f((float)i - 0.1, (float)j, (float)k + 0.1);
			glVertex3f((float)i + 0.1, (float)j, (float)k + 0.1);
			glVertex3f((float)i + 0.1, (float)j, (float)k - 0.1);
			glEnd();
			glBegin(GL_QUADS);
			glVertex3f((float)i, (float)j - 0.1, (float)k - 0.1);
			glVertex3f((float)i, (float)j - 0.1, (float)k + 0.1);
			glVertex3f((float)i, (float)j + 0.1, (float)k + 0.1);
			glVertex3f((float)i, (float)j + 0.1, (float)k - 0.1);
			glEnd();

		}

	}

}


void LBM::renderheat_plane()
{
	int i, j, k;

	float color[3] = { 0, 0, 0 };
	float tmpcolor;
	for (i = 0; i < SIZE_3D_X; i++)
	for (j = 0; j < SIZE_3D_Y; j++)
	for (k = 0; k < SIZE_3D_Z; k++)
	{
		tmpcolor = (h_T[getid(i, j, k)] - T0) / 10.0;
		//float tmpcolor = (lbm.T[i][j][k]*1.0);
		if (tmpcolor < 0)
			color[2] = 1;

		int ic = (int)tmpcolor;
		float f = tmpcolor - ic;
		switch (ic)
		{
		case 0:
		{
				  color[0] = 0;
				  color[1] = f / 2;
				  color[2] = 1;
		}
			break;
		case 1:
		{

				  color[0] = 0;
				  color[1] = f / 2 + 0.5f;
				  color[2] = 1;
		}
			break;
		case 2:
		{
				  color[0] = f / 2;
				  color[1] = 1;
				  color[2] = 1 - f / 2;
		}
			break;
		case 3:
		{
				  color[0] = f / 2 + 0.5f;
				  color[1] = 1;
				  color[2] = 0.5f - f / 2;
		}
			break;
		case 4:
		{
				  color[0] = 1;
				  color[1] = 1.0f - f / 2;
				  color[2] = 0;
		}
			break;
		case 5:
		{
				  color[0] = 1;
				  color[1] = 0.5f - f / 2;
				  color[2] = 0;
		}
			break;
		default:
		{
				   color[0] = 1;
				   color[1] = 0;
				   color[2] = 0;
		}
			break;
		}

		{
			glColor4f(color[0], color[1], color[2], 0.5);//设置颜色
			// 			glBegin(GL_QUADS);
			// 			glVertex3f((float)i - 0.1, (float)j - 0.1, (float)k);
			// 			glVertex3f((float)i - 0.1, (float)j + 0.1, (float)k);
			// 			glVertex3f((float)i + 0.1, (float)j + 0.1, (float)k);
			// 			glVertex3f((float)i + 0.1, (float)j - 0.1, (float)k);
			// 			glEnd();
			glBegin(GL_QUADS);
			glVertex3f((float)i - 0.4, (float)0.75*SIZE_3D_Y, (float)k - 0.4);
			glVertex3f((float)i - 0.4, (float)0.75*SIZE_3D_Y, (float)k + 0.4);
			glVertex3f((float)i + 0.4, (float)0.75*SIZE_3D_Y, (float)k + 0.4);
			glVertex3f((float)i + 0.4, (float)0.75*SIZE_3D_Y, (float)k - 0.4);
			glEnd();
			// 			glBegin(GL_QUADS);
			// 			glVertex3f((float)i, (float)j - 0.1, (float)k - 0.1);
			// 			glVertex3f((float)i, (float)j - 0.1, (float)k + 0.1);
			// 			glVertex3f((float)i, (float)j + 0.1, (float)k + 0.1);
			// 			glVertex3f((float)i, (float)j + 0.1, (float)k - 0.1);
			// 			glEnd();

		}

	}

}

void LBM::render_vel()
{
	int i, j, k;

	real rho = 0;
	vec3_t vel = { 0 };

	float color[3] = { 0, 0, 0 };
	float tmppos[3];
	float len = 0.0;
	for (i = 0; i < SIZE_3D_X; i++)
		for (j = 0; j < SIZE_3D_Y; j++)
			for (k = 0; k < SIZE_3D_Z; k++)
	{
		///////////////////////////////////////////
		int x, y, z;

		
		
					// current cell
		const LBM::dfD3Q19_t *df = LBM3DField_Get(&fieldevolv[frame], i, j, k);

				//	rho += df->rho;

					// weighted contribution to overall velocity
// 					vel.x += df->rho * df->u.x;
// 					vel.y += df->rho * df->u.y;
// 					vel.z += df->rho * df->u.z;
	
		//////////////////////////////////////////
		//len = (h_vx[getid(i, j, k)] * h_vx[getid(i, j, k)] + h_vy[getid(i, j, k)] * h_vy[getid(i, j, k)] + h_vz[getid(i, j, k)] * h_vz[getid(i, j, k)]);
			len = df->u.x*df->u.x + df->u.y*df->u.y + df->u.z*df->u.z;
		// 		tmppos[0] = h_vx[getid(i, j, k)] * len * 1000;
		// 		tmppos[1] = h_vy[getid(i, j, k)] * len * 1000;
		// 		tmppos[2] = h_vz[getid(i, j, k)] * len * 1000;
		tmppos[0] = (i - 0.5) / SIZE_3D_X;
		tmppos[1] = (j - 0.5) / SIZE_3D_Y;
		tmppos[2] = (k - 0.5) / SIZE_3D_Z;


		float tmpcolor = len * 200;
		if (tmpcolor < 0)
			color[2] = 0;

		int ic = (int)tmpcolor;
		float f = tmpcolor - ic;
		switch (ic)
		{
		case 0:
		{
				  color[0] = 0;
				  color[1] = f / 2;
				  color[2] = 0;
		}
			break;
		case 1:
		{

				  color[0] = 0;
				  color[1] = f / 2 + 0.5f;
				  color[2] = 0;
		}
			break;
		case 2:
		{
				  color[0] = f / 2;
				  color[1] = 1;
				  color[2] = 1 - f / 2;
		}
			break;
		case 3:
		{
				  color[0] = f / 2 + 0.5f;
				  color[1] = 1;
				  color[2] = 0.5f - f / 2;
		}
			break;
		case 4:
		{
				  color[0] = 1;
				  color[1] = 1.0f - f / 2;
				  color[2] = 0;
		}
			break;
		case 5:
		{
				  color[0] = 1;
				  color[1] = 0.5f - f / 2;
				  color[2] = 0;
		}
			break;
		default:
		{
				   color[0] = 1;
				   color[1] = 0;
				   color[2] = 0;
		}
			break;
		}

		{
			glColor4f(color[0], color[1], color[2], 0.5);//设置颜色
			 			glBegin(GL_QUADS);
			 			glVertex3f((float)i - 0.1, (float)j - 0.1, (float)k);
			 			glVertex3f((float)i - 0.1, (float)j + 0.1, (float)k);
			 			glVertex3f((float)i + 0.1, (float)j + 0.1, (float)k);
			 			glVertex3f((float)i + 0.1, (float)j - 0.1, (float)k);
			 			glEnd();
// 			glBegin(GL_LINES);
// 			glVertex3f((float)i, (float)j, (float)k);
// 			glVertex3f((float)i + h_vx[getid(i, j, k)] * 5, (float)j + h_vy[getid(i, j, k)] * 5, (float)k + h_vz[getid(i, j, k)] * 5);
// 			glEnd();
			// 			glBegin(GL_QUADS);
			// 			glVertex3f((float)i, (float)j - 0.1, (float)k - 0.1);
			// 			glVertex3f((float)i, (float)j - 0.1, (float)k + 0.1);
			// 			glVertex3f((float)i, (float)j + 0.1, (float)k + 0.1);
			// 			glVertex3f((float)i, (float)j + 0.1, (float)k - 0.1);
			// 			glEnd();

		}

	}

}

void LBM::render_dens()
{
	int i, j, k;

	float color[3] = { 0, 0, 0 };
	float tmpcolor;
	for (i = 1; i < SIZE_3D_X; i++)
	for (j = 1; j < SIZE_3D_Y; j++)
	for (k = 1; k < SIZE_3D_Z; k++)
	{
		tmpcolor = (h_rho[getid(i, j, k)] - 0.9*RHO) * 3;
		//float tmpcolor = (lbm.T[i][j][k]*1.0);
		if (tmpcolor < 0)
			color[2] = 1;

		int ic = (int)tmpcolor;
		float f = tmpcolor - ic;
		switch (ic)
		{
		case 0:
		{
				  color[0] = 0;
				  color[1] = f / 2;
				  color[2] = 1;
		}
			break;
		case 1:
		{

				  color[0] = 0;
				  color[1] = f / 2 + 0.5f;
				  color[2] = 1;
		}
			break;
		case 2:
		{
				  color[0] = f / 2;
				  color[1] = 1;
				  color[2] = 1 - f / 2;
		}
			break;
		case 3:
		{
				  color[0] = f / 2 + 0.5f;
				  color[1] = 1;
				  color[2] = 0.5f - f / 2;
		}
			break;
		case 4:
		{
				  color[0] = 1;
				  color[1] = 1.0f - f / 2;
				  color[2] = 0;
		}
			break;
		case 5:
		{
				  color[0] = 1;
				  color[1] = 0.5f - f / 2;
				  color[2] = 0;
		}
			break;
		default:
		{
				   color[0] = 1;
				   color[1] = 0;
				   color[2] = 0;
		}
			break;
		}

		{
			glColor4f(color[0], color[1], color[2], 0.5);//设置颜色
			// 			glBegin(GL_QUADS);
			// 			glVertex3f((float)i - 0.1, (float)j - 0.1, (float)k);
			// 			glVertex3f((float)i - 0.1, (float)j + 0.1, (float)k);
			// 			glVertex3f((float)i + 0.1, (float)j + 0.1, (float)k);
			// 			glVertex3f((float)i + 0.1, (float)j - 0.1, (float)k);
			// 			glEnd();
			glBegin(GL_QUADS);
			glVertex3f((float)i - 0.45, (float)0.75*SIZE_3D_Y, (float)k - 0.45);
			glVertex3f((float)i - 0.45, (float)0.75*SIZE_3D_Y, (float)k + 0.45);
			glVertex3f((float)i + 0.45, (float)0.75*SIZE_3D_Y, (float)k + 0.45);
			glVertex3f((float)i + 0.45, (float)0.75*SIZE_3D_Y, (float)k - 0.45);
			glEnd();
			// 			glBegin(GL_QUADS);
			// 			glVertex3f((float)i, (float)j - 0.1, (float)k - 0.1);
			// 			glVertex3f((float)i, (float)j - 0.1, (float)k + 0.1);
			// 			glVertex3f((float)i, (float)j + 0.1, (float)k + 0.1);
			// 			glVertex3f((float)i, (float)j + 0.1, (float)k - 0.1);
			// 			glEnd();

		}

	}

}

void LBM::outputuxyz()
{

	if (frame % 10 == 0)
		printf("outputfile:%i  center T:%f V:%f: total E:%f \n", frame, h_T[getid(SIZE_3D_X / 2, SIZE_3D_Y / 2, SIZE_3D_Z / 2)], h_vz[getid(SIZE_3D_X / 8, SIZE_3D_Y / 2, SIZE_3D_Z / 2)], frame/*dafdfasdfasfasdfasfsdfasfdfasdfeeee*/);
	if (frame % 1000 == 0 && frame != 0)
	{
		int i, j, k;

		ostringstream name;
		name << "test" << frame << ".dat";
		ofstream out(name.str().c_str());
		out << "Title=\"DDF-LBM Flow\"\n"
			<< "VARIABLES=\"X\",\"Y\",\"Z\",\"U\",\"V\",\"T\"\n" << endl;
		/*	out <<double(i)/Lx<<" "
		<<double(j)/Ly<<" "
		<<u[i][j][0]<<" "
		<<u[i][j][1]<<  endl;*/
		for (i = 0; i < SIZE_3D_X; i++)
		for (j = 0; j < SIZE_3D_Y; j++)
		for (k = 0; k < SIZE_3D_Z; k++)
		for (int a = 0; a < 19; a++)
		{
			out << i << "   "
				<< j << "   "
				<< k << "     " << a << "               "//<< F[i][j][k][a] << "        " << H[i][j][k][a]
				<< h_vx[getid(i, j, k)] << "   "
				<< h_vy[getid(i, j, k)] << "   "
				<< h_vz[getid(i, j, k)] << "  rho: "
				<< h_rho[getid(i, j, k)] << "   "
				<< h_f[getid(i, j, k)] << "   "
				<< h_h[getid(i, j, k)]
				<< endl;
		}

	}
	frame++;
}

void LBM::rollrendermode()
{
	rendermode = (ERENDERMODE)((rendermode + 1) % RENDER_CNT);
	printf("rendermode=%d\n", (int)rendermode);
}