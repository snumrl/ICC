#include "SimWindow.h"
#include <vector>
#include <string>
#include <GL/glut.h>

int main(int argc,char** argv)
{
	if( argc < 2 ) {
		std::cout << "Please input a filename" << std::endl;
		return 0;
	}

	// std::cout<<"[ : Frame --"<<std::endl;
	// std::cout<<"] : Frame ++"<<std::endl;
	// std::cout<<"r : Frame = 0"<<std::endl;
	// std::cout<<"C : Capture"<<std::endl;
	// std::cout<<"SPACE : Play"<<std::endl;
	// std::cout<<"ESC : exit"<<std::endl;
	SimWindow* simwindow = new SimWindow(std::string(argv[1]));

	glutInit(&argc, argv);
	simwindow->InitWindow(1600,900,"Render");
	glutMainLoop();
}
