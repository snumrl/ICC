#ifndef __GUI_GLUT_WINDOW_H__
#define __GUI_GLUT_WINDOW_H__
#include <GL/glew.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <memory>
#include <iostream>

#define GLSL(version, shader)  "#version " #version "\n" #shader
// std::string vert;
// char vert[] = GLSL(120,
// varying vec3 N;
// varying vec3 v;
// void main(void)
// {
//   v = vec3(gl_ModelViewMatrix * gl_Vertex);
//   N = normalize(gl_NormalMatrix * gl_Normal);
//   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
// }
// );
// const char frag[] = GLSL(120,
// varying vec3 vN;
// varying vec3 v; 

// #define MAX_LIGHTS 3

// void main (void) 
// { 
//    vec3 N = normalize(vN);
//    vec4 finalColor = vec4(0.0, 0.0, 0.0, 0.0);
//    for (int i=0;i<MAX_LIGHTS;i++)
//    {
//       vec3 L = normalize(gl_LightSource[i].position.xyz - v); 
//       vec3 E = normalize(-v); // we are in Eye Coordinates, so EyePos is (0,0,0) 
//       vec3 R = normalize(-reflect(L,N)); 
   
//       //calculate Ambient Term: 
//       vec4 Iamb = gl_FrontLightProduct[i].ambient; 
//       //calculate Diffuse Term: 
//       vec4 Idiff = gl_FrontLightProduct[i].diffuse * max(dot(N,L), 0.0);
//       Idiff = clamp(Idiff, 0.0, 1.0); 
   
//       // calculate Specular Term:
//       vec4 Ispec = gl_FrontLightProduct[i].specular 
//              * pow(max(dot(R,E),0.0),0.3*gl_FrontMaterial.shininess);
//       Ispec = clamp(Ispec, 0.0, 1.0); 
   
//       finalColor += Iamb + Idiff + Ispec;
//    }
   
//    // write Total Color: 
//    gl_FragColor = gl_FrontLightModelProduct.sceneColor + finalColor; 
// }
// );

namespace GUI
{
class Camera;
class GLUTWindow
{
public:
//Rendering Part 
GLuint program = 0, vertShader = 0, fragShader = 0;
void printShaderInfoLog(GLuint shader) {
  int len = 0;
  int charsWritten = 0;
  char* infoLog;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
  if (len > 0) {
    infoLog = (char*)malloc(len);
    glGetShaderInfoLog(shader, len, &charsWritten, infoLog);
    printf("%s\n", infoLog);
    free(infoLog);
  }
}
void printProgramInfoLog(GLuint obj) {
  int len = 0, charsWritten = 0;
  char* infoLog;
  glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &len);
  if (len > 0) {
    infoLog = (char*)malloc(len);
    glGetProgramInfoLog(obj, len, &charsWritten, infoLog);
    printf("%s\n", infoLog);
    free(infoLog);
  }
}

GLuint createShader(const char* src, GLenum type) {
  GLuint shader;
  shader = glCreateShader(type);
  glShaderSource(shader, 1, &src, NULL);
  glCompileShader(shader);
  printShaderInfoLog(shader);
  return shader;
}

void createProgram(){
  std::string vertex_file = std::string(DPHY_DIR)+std::string("/renderer/vertexshader.txt");
  FILE *f = fopen(vertex_file.c_str(), "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  char *vert = (char *)malloc(fsize + 1);
  fread(vert, fsize, 1, f);
  fclose(f);

  vert[fsize] = 0;

  std::string frag_file = std::string(DPHY_DIR)+std::string("/renderer/fragshader.txt");
  f = fopen(frag_file.c_str(), "rb");
  fseek(f, 0, SEEK_END);
  fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  char *frag = (char *)malloc(fsize + 1);
  fread(frag, fsize, 1, f);
  fclose(f);

  frag[fsize] = 0;


  vertShader = createShader(vert, GL_VERTEX_SHADER);
  fragShader = createShader(frag, GL_FRAGMENT_SHADER);
  program = glCreateProgram();
  glAttachShader(program, vertShader);
  glAttachShader(program, fragShader);
  glLinkProgram(program);
  printProgramInfoLog(program);
}

void initGL() {
  GLenum err = glewInit();
  if (err != GLEW_OK) {
    printf("GLEW init failed.\n");
    printf("Error : %s\n", glewGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  createProgram();
}
public:
	GLUTWindow();
	~GLUTWindow();

	virtual void InitWindow(int _w,int _h,const char* _name);
	
	static GLUTWindow* current();
	static void DisplayEvent();
	static void KeyboardEvent(unsigned char key,int x,int y);
	static void MouseEvent(int button, int state, int x, int y);
	static void MotionEvent(int x, int y);
	static void ReshapeEvent(int w, int h);
	static void TimerEvent(int value);

	static std::vector<GLUTWindow*> mWindows;
	static std::vector<int> mWinIDs;
	
protected:
	virtual void initLights(double x = 0.0, double z = 0.0, double fx = 0.0, double fz = 0.0);
	virtual void Display() = 0;
	virtual void Keyboard(unsigned char key,int x,int y) = 0;
	virtual void Mouse(int button, int state, int x, int y) = 0;
	virtual void Motion(int x, int y) = 0;
	virtual void Reshape(int w, int h) = 0;
	virtual void Timer(int value) = 0;
protected:
	std::unique_ptr<Camera> 		mCamera;
	bool 							mIsDrag;
	int 							mMouseType;
	int 							mPrevX,mPrevY;
	int 							mDisplayTimeout;
		  std::vector<unsigned char> mScreenshotTemp;
  std::vector<unsigned char> mScreenshotTemp2;
};

};
#endif