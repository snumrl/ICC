#include "InteractiveWindow.h"
#include <vector>
#include <string>
#include <GL/glut.h>

int main(int argc,char** argv)
{
	if( argc < 2 ) {
		std::cout << "Please input a filename" << std::endl;
		return 0;
	}
	InteractiveWindow* interactiveWindow;
	if( argc == 3)
		interactiveWindow = new InteractiveWindow(std::string(argv[1]), std::string(argv[2]));
	else
		interactiveWindow = new InteractiveWindow(std::string(argv[1]));

	glutInit(&argc, argv);
	interactiveWindow->initWindow(1920,1080,"Interactive");
	glutMainLoop();
}
