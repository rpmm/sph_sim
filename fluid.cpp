#include <GL/glut.h>
#include <iostream>
#include <vector>
#include "eigen3/Eigen/Dense"
#include <math.h> // sin,cos

using namespace std;
using namespace Eigen;

// g++ fluid.cpp -o fluidsim -lGLU -lGL -lglut -O2 -msse2 -DNDEBUG -fopenmp


// Solver Parameters
const static Vector2f g(0.f, -9.81f);		// Gravitational acceleration (m/s^2)
const static float spacing = 9.f;			// Spacing (m)
const static float rho_0 = 1000.f;			// Initial density (kg/m^3)
const static float k_gasconst = 30.f;		// Resistance to compression (J)
const static float mu = 25.f;				// Viscosity (Pa s * 10^-3)
const static float m = 100.f;				// Assume all particles have the same mass (kg)
const static float damping = -0.75f;		// Coefficient of restitution
const static float dt = 0.025f;

// Smoothing kernels
const static float h = spacing;				// Smoothing length
const static float h2 = pow(h, 2.f);		// Optimization
const static float W_poly6 = 315.f / (64.f * M_PI * pow(h, 9.f));
const static float W_spiky = -45.f / (M_PI * pow(h, 6.f)); // Gradient
const static float visc_laplacian = 45.f / (M_PI * pow(h, 6.f)); // Laplacian

// Domain, Window, Dam
const static float window_width = 1000;
const static float window_height = 600;
const static int dam_width = window_width * 0.45f;
const static int dam_height = window_height * 0.95f;

// Misc
int debug_1 = 0;

// Mouse interraction
Vector2f mouse_pos = {99999,99999}; // Initial cursor position
bool mouse_clicked;

// Particle Structure
struct Particle {
	Vector2f pos, vel, acc, force;
	Vector2f pos_h, pos_old, vel_h;
	float rho, p;
	vector<Particle> neighbors;
	Particle() // Constructor
	{
		pos = {0.f, 0.f}; pos_h = {0.f, 0.f};
		vel = {0.f, 0.f}; vel_h = {0.f, 0.f};
		acc = {0.f, 0.f}; force = {0.f, 0.f};
		rho = 1.f; // Avoid division by zero
		p = 0.f;
	}
};

static vector<Particle> particles;





void InitSPH(void)
{
	for(float y = h; y <= dam_height; y += spacing)
		for(float x = h; x <= dam_width; x += spacing)
		{
			float rand01 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/0.01f); // 0 to 0.01
			Particle newparticle;
			newparticle.pos = {x+rand01,y+rand01};
			particles.push_back(newparticle);
		}
	cout << "Particle amount: " << particles.size() << endl;
}

void DensityAndPressure(void)
{
	#pragma omp parallel for
	for(auto &p_i : particles)
	{
		p_i.rho = 1.f;
		p_i.neighbors.clear();

		for(auto &p_j : particles)
		{
			float r2 = (p_j.pos - p_i.pos).squaredNorm();

			if(r2 <= h2){
				p_i.rho += m * W_poly6 * pow(h2 - r2, 3.f); // Density
				
				// Add particle to neighborhood
				Particle neighbor;
				neighbor.pos = p_j.pos; neighbor.p = p_j.p;
				neighbor.rho = p_j.rho; neighbor.vel = p_j.vel;
				p_i.neighbors.push_back(neighbor);
			}
		}
		p_i.p = k_gasconst * (p_i.rho - rho_0); // Pressure
	}
}

void Forces(void)
{
	#pragma omp parallel for
	for(auto &p_i : particles)
	{
		Vector2f f_pressure(0.f, 0.f), f_viscosity(0.f, 0.f);

		for(auto &p_j : p_i.neighbors)
		{
			Vector2f r_ij = p_j.pos - p_i.pos;
			float r = r_ij.norm();
			f_pressure += -r_ij.normalized() * m * (p_i.p + p_j.p) / (2.f * p_j.rho) * W_spiky * pow(h - r, 2.f);
			f_viscosity += mu * m * (p_j.vel - p_i.vel) / p_j.rho * visc_laplacian * (h - r);
		}

		p_i.acc = (f_pressure + f_viscosity + g * p_i.rho) / p_i.rho;

		// Attraction forces to mouse location
		float dist_to_mouse = (p_i.pos - mouse_pos).norm();
		if(mouse_clicked)
			if(dist_to_mouse < window_width * 0.05f) // Threshold for which particles are attracted
				p_i.acc += (p_i.pos - mouse_pos) * 2; // Scalar to scale the mouse force
	}
}

// void LeapFrogInit(void)
// {
// 	for(auto &p : particles)
// 	{
// 		p.vel_h(0.f, 0.f), p.vel(0.f, 0.f), p.pos(0.f,0.f);
// 		p.vel_h = p.vel + p.acc * dt * 0.5f; // update half step velocity
// 		p.vel +=  p.acc * dt; // update position
// 		p.pos += p.vel_h * dt;	// update velocity

// 		if(p.pos(0) < 0) { p.vel(0) *= damping; p.pos(0) = 0; } // Left
// 		if(p.pos(1) < 0) { p.vel(1) *= damping; p.pos(1) = 0; } // Bottom
// 		if(p.pos(0) > window_width-h) { p.vel(0) *= damping; p.pos(0) = window_width-h;  }  // Right
// 		//if(p.pos(1) > domain_height){ p.vel(1) *= damping; p.pos(1) = domain_height; } // Top
// 	}
// }


void Integrate(void)
{
	#pragma omp parallel for
	for(auto &p : particles) // Loop through all particles
	{
		// Forward Euler integration
		p.vel += p.acc * dt;
		p.pos += p.vel * dt;

		// Leap Frog
		// p.vel_h += p.acc * dt; // update half step velocity
		// p.vel = p.vel_h + p.acc * dt/2;	// update velocity
		// p.pos +=  p.vel_h * dt; // update position
	
		// Boundaries
		if(p.pos(0) < h) { p.vel(0) *= damping; p.pos(0) = h; } // Left
		if(p.pos(1) < h) { p.vel(1) *= damping; p.pos(1) = h; } // Bottom
		if(p.pos(0) > window_width-h) { p.vel(0) *= damping; p.pos(0) = window_width-h;  }  // Right
		if(p.pos(1) > window_height*2){ p.vel(1) *= damping; p.pos(1) = window_height*2; } // Top

		// Debugging
		if(fabs(p.vel.norm()) > 90)
			p.vel *= 0.75f;
	}
}

void Update(void)
{
	DensityAndPressure();
	Forces();
	Integrate();
	glutPostRedisplay(); // Marks the current window as needing to be redisplayed
}

void Mouse_Pos(int x, int y)
{
	Vector2f mouse = Vector2f(x,fabs(y-window_height)); // Updates the location of the mouse
	mouse_pos = mouse;
}


void Mouse(int button, int state, int x, int y)
{
	if(state == GLUT_DOWN)
		mouse_clicked = true;
	else
	{
		mouse_clicked = false;
		mouse_pos = Vector2f(window_height * 100, window_width * 100);
	}
}

void Keyboard(unsigned char c, __attribute__((unused)) int pos, __attribute__((unused)) int y)
{   
	int origo_x = window_width * 0.5f;
	int origo_y = window_height * 0.7f;
	switch(c)
	{
		case 'b': // Drop a box
			for(float y = window_height * 0.6f; y < window_height * 0.8f; y += h)
				for(float x = window_width * 0.4f; x <  window_width * 0.6f; x += h)
				{
					Particle newparticle;
					newparticle.pos = {x, y};
					particles.push_back(newparticle);
				}
			cout << "Particle amount: " << particles.size() << endl;
			break;

		case 's': // Drop a sphere
			for(float radius = spacing * 1.05f; radius < 50; radius += spacing)
				{
				for(float theta = 0.f; theta <= 2*M_PI - spacing/radius; theta += spacing * 1.05f / radius)
					{
						Particle newparticle;
						newparticle.pos = {origo_x + cos(theta) * radius, origo_y + sin(theta) * radius};
						particles.push_back(newparticle);
					}
				}
			cout << "Particle amount: " << particles.size() << endl;
			break;

		case 'r':
			particles.clear();
			InitSPH();
			break;

		case 'f':
			debug_1 = 1 - debug_1;
			cout << "Debug mode: " << debug_1 << endl;
			break;

		case 27: // Esc
			cout << "Program stopped!" << endl;
			exit(0);

	}
}

void InitGL(void)
{
	glClearColor(1.f, 1.f, 1.f, 1);	// Window background color
	glEnable(GL_POINT_SMOOTH);			// Anti-aliasing
	glPointSize(spacing*0.55f);			// Point size	
	glMatrixMode(GL_PROJECTION);		// View projection matrix
}

void Render(void)
{
	glClear(GL_COLOR_BUFFER_BIT);					// Blank image buffer state
	glLoadIdentity();								// Replace the current matrix with the identity matrix
	gluOrtho2D(0, window_width, 0, window_height);	// Orthographic projection
	glColor3f(0.2f, 0.6f, 1.0f);					// Particle color
	glBegin(GL_POINTS);								// Start drawing vertices
	for(Particle &p : particles){
		float col1 = p.vel.squaredNorm() * 0.001f;
		float col2 = fabs(p.p * 0.000002);
		glColor3f(col1, 0, 1-col1);
		glVertex2f(p.pos(0), p.pos(1));				// Place vertex at particle coordinate
		
	}
	glEnd();										// Stop drawing vertices
	glutSwapBuffers();								// Swaps the buffers of the current window if double buffered
}


int main(int argc, char** argv)
{
	glutInitWindowSize(window_width, window_height);// Window is initialized
	glutInit(&argc, argv);							// Initialize the GLUT library
	glutCreateWindow("2D SPH - CFM Project");		// Creates a top-level window
	glutDisplayFunc(Render);						// Generate a new frame
	glutIdleFunc(Update);							// Time between rendering
	
	glutKeyboardFunc(Keyboard);						// Sets the keyboard callback for the current window
	glutMotionFunc(Mouse_Pos);
	glutMouseFunc(Mouse);

	InitGL();										// Initalize rendering environment
	InitSPH();
	glutMainLoop();									// Enters the GLUT event processing loop
	
	return 0;
}