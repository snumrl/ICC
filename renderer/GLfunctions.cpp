#include "GLfunctions.h"
#include <assimp/cimport.h>
#include <iostream>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include "GL/glut.h"

#define USE_VNORMAL 0
#define RESOLUTION 16

static GLUquadricObj *quadObj;
static void initQuadObj(void)
{
    quadObj = gluNewQuadric();
    if(!quadObj)
        // DART modified error output
        std::cerr << "OpenGL: Fatal Error in DART: out of memory." << std::endl;
}
#define QUAD_OBJ_INIT { if(!quadObj) initQuadObj(); }
void
GUI::
DrawSphere(double r)
{
    QUAD_OBJ_INIT;
    gluQuadricDrawStyle(quadObj, GLU_FILL);
    gluQuadricNormals(quadObj, GLU_SMOOTH);

    gluSphere(quadObj, r, RESOLUTION, RESOLUTION);
}
//==============================================================================
void
GUI::
DrawOpenDome(double radius, int slices, int stacks)
{
  // (2pi/Stacks)
  const auto pi = M_PI;
  const auto drho = pi / stacks / 2.0;
  const auto dtheta = 2.0 * pi / slices;

  const auto rho = drho;
  const auto srho = std::sin(rho);
  const auto crho = std::cos(rho);

  // Many sources of OpenGL sphere drawing code uses a triangle fan
  // for the caps of the sphere. This however introduces texturing
  // artifacts at the poles on some OpenGL implementations
  glBegin(GL_TRIANGLE_FAN);
  glNormal3d(0.0, 0.0, radius);
  glVertex3d(0.0, 0.0, radius);
  for (int j = 0; j <= slices; ++j)
  {
    const auto theta = (j == slices) ? 0.0 : j * dtheta;
    const auto stheta = -std::sin(theta);
    const auto ctheta = std::cos(theta);

    const auto x = srho * stheta;
    const auto y = srho * ctheta;
    const auto z = crho;

    glNormal3d(x, y, z);
    glVertex3d(x * radius, y * radius, z * radius);
  }
  glEnd();

  for (int i = 1; i < stacks; ++i)
  {
    const auto rho = i * drho;
    const auto srho = std::sin(rho);
    const auto crho = std::cos(rho);
    const auto srhodrho = std::sin(rho + drho);
    const auto crhodrho = std::cos(rho + drho);

    // Many sources of OpenGL sphere drawing code uses a triangle fan
    // for the caps of the sphere. This however introduces texturing
    // artifacts at the poles on some OpenGL implementations
    glBegin(GL_TRIANGLE_STRIP);

    for (int j = 0; j <= slices; ++j)
    {
      const auto theta = (j == slices) ? 0.0 : j * dtheta;
      const auto stheta = -std::sin(theta);
      const auto ctheta = std::cos(theta);

      auto x = srho * stheta;
      auto y = srho * ctheta;
      auto z = crho;

      glNormal3d(x, y, z);
      glVertex3d(x * radius, y * radius, z * radius);

      x = srhodrho * stheta;
      y = srhodrho * ctheta;
      z = crhodrho;

      glNormal3d(x, y, z);
      glVertex3d(x * radius, y * radius, z * radius);
    }
    glEnd();
  }
}


void
GUI::
DrawOpenDomeQuater(double radius, int slices, int stacks)
{
    // (2pi/Stacks)
    const auto pi = M_PI;
    const auto drho = pi / stacks / 2.0; // x, y
    const auto dtheta = 0.5 * pi / slices; // z

    const auto rho = drho;
    const auto srho = std::sin(rho);
    const auto crho = std::cos(rho);

    // Many sources of OpenGL sphere drawing code uses a triangle fan
    // for the caps of the sphere. This however introduces texturing
    // artifacts at the poles on some OpenGL implementations
    glBegin(GL_TRIANGLE_FAN);
    glNormal3d(0.0, 0.0, radius);
    glVertex3d(0.0, 0.0, radius);
    for (int j = 0; j <= slices; ++j)
    {
        const auto theta = j * dtheta;
        const auto stheta = -std::sin(theta);
        const auto ctheta = std::cos(theta);

        const auto x = srho * stheta;
        const auto y = srho * ctheta;
        const auto z = crho;

        glNormal3d(x, y, z);
        glVertex3d(x * radius, y * radius, z * radius);
    }
    glEnd();

    for (int i = 1; i < stacks; ++i)
    {
        const auto rho = i * drho;
        const auto srho = std::sin(rho);
        const auto crho = std::cos(rho);
        const auto srhodrho = std::sin(rho + drho);
        const auto crhodrho = std::cos(rho + drho);

        // Many sources of OpenGL sphere drawing code uses a triangle fan
        // for the caps of the sphere. This however introduces texturing
        // artifacts at the poles on some OpenGL implementations
        glBegin(GL_TRIANGLE_STRIP);

        for (int j = 0; j <= slices; ++j)
        {
            const auto theta = j * dtheta;
            const auto stheta = -std::sin(theta);
            const auto ctheta = std::cos(theta);

            auto x = srho * stheta;
            auto y = srho * ctheta;
            auto z = crho;

            glNormal3d(x, y, z);
            glVertex3d(x * radius, y * radius, z * radius);

            x = srhodrho * stheta;
            y = srhodrho * ctheta;
            z = crhodrho;

            glNormal3d(x, y, z);
            glVertex3d(x * radius, y * radius, z * radius);
        }
        glEnd();
    }
}



void
GUI::
DrawCapsule(double radius, double height)
{
    GLint slices = RESOLUTION;
    GLint stacks = RESOLUTION;

    // Graphics assumes Cylinder is centered at CoM
    // gluCylinder places base at z = 0 and top at z = height
    glTranslated(0.0, 0.0, -0.5*height);

    // Code taken from glut/lib/glut_shapes.c
    QUAD_OBJ_INIT;
    gluQuadricDrawStyle(quadObj, GLU_FILL);
    gluQuadricNormals(quadObj, GLU_SMOOTH);

    gluCylinder(quadObj, radius, radius, height, slices, stacks); //glut/lib/glut_shapes.c

    // Upper hemisphere
    glTranslated(0.0, 0.0, height);
    DrawOpenDome(radius, slices, stacks);

    // Lower hemisphere
    glTranslated(0.0, 0.0, -height);
    glRotated(180.0, 0.0, 1.0, 0.0);
    DrawOpenDome(radius, slices, stacks);
}

void
GUI::
DrawCylinder(double radius, double height)
{
    GLint slices = RESOLUTION;
    GLint stacks = RESOLUTION;

    // Graphics assumes Cylinder is centered at CoM
    // gluCylinder places base at z = 0 and top at z = height
    glTranslated(0.0, 0.0, -0.5*height);

    // Code taken from glut/lib/glut_shapes.c
    QUAD_OBJ_INIT;
    gluQuadricDrawStyle(quadObj, GLU_FILL);
    gluQuadricNormals(quadObj, GLU_SMOOTH);

    gluCylinder(quadObj, radius, radius, height, slices, stacks); //glut/lib/glut_shapes.c
}

GLfloat Abs(GLfloat x){
    return x>0?x:-x;
}

void moveToCenter(GLfloat from[3], GLfloat to[3], double radius ,GLfloat res[3]) {
    GLfloat vec[3];
    for (int i = 0; i < 3; i++) vec[i] = to[i] - from[i];
    GLfloat minV = Abs(vec[0]);
    int minIdx = 0;
    for (int i = 1; i < 3; i++) {
        if (Abs(vec[i]) < minV) {
            minV = Abs(vec[i]);
            minIdx = i;
        }
    }

    for (int i = 0; i < 3; i++) {
        if (minIdx == i) vec[i] = 0;
        else {
            if (vec[i] > 0) vec[i] = 1;
            else vec[i] = -1;
        }
        vec[i] *= radius;
    }
    for (int i = 0; i < 3; i++) {
        res[i] = from[i] + vec[i];
    }
}

void GUI::DrawRoundedBox(const Eigen::Vector3d& size, double radius){
    DrawRoundedBoxPlanes(size, radius);

    DrawRoundedBoxCylinder(Eigen::Vector2d(size[0], size[2]), size[1] - radius * 2, radius);
    glPushMatrix();
    glRotated(90, 0, 0, 1);
    DrawRoundedBoxCylinder(Eigen::Vector2d(size[1], size[2]), size[0] - radius * 2, radius);
    glPopMatrix();
    glPushMatrix();
    glRotated(90, 1, 0, 0);
    DrawRoundedBoxCylinder(Eigen::Vector2d(size[0], size[1]), size[2] - radius * 2, radius);
    glPopMatrix();

    DrawRoundedBoxSphere(size, radius);
}
void GUI::DrawRoundedBox(const Eigen::Vector3d& size, double radius, const Eigen::Vector3d& uniform_color){
    DrawRoundedBoxPlanes(size, radius);

    glPushMatrix();
    {
        glColor3f(uniform_color[0],uniform_color[1],uniform_color[2]);
        DrawRoundedBoxCylinder(Eigen::Vector2d(size[0], size[2]), size[1] - radius * 2, radius);
        glPushMatrix();
        glRotated(90, 0, 0, 1);
        DrawRoundedBoxCylinder(Eigen::Vector2d(size[1], size[2]), size[0] - radius * 2, radius);
        glPopMatrix();
        glPushMatrix();
        glRotated(90, 1, 0, 0);
        DrawRoundedBoxCylinder(Eigen::Vector2d(size[0], size[1]), size[2] - radius * 2, radius);
        glPopMatrix();

        DrawRoundedBoxSphere(size, radius);
    }
    glPopMatrix();
}

void GUI::DrawRoundedBoxSphere(const Eigen::Vector3d& size, double radius){
    GLfloat v[8][3];

    v[0][0] = v[1][0] = v[2][0] = v[3][0] = -size[0] / 2 + radius;
    v[4][0] = v[5][0] = v[6][0] = v[7][0] = size[0] / 2 - radius;
    v[0][1] = v[1][1] = v[4][1] = v[5][1] = -size[1] / 2 + radius;
    v[2][1] = v[3][1] = v[6][1] = v[7][1] = size[1] / 2 - radius;
    v[0][2] = v[3][2] = v[4][2] = v[7][2] = -size[2] / 2 + radius;
    v[1][2] = v[2][2] = v[5][2] = v[6][2] = size[2] / 2 - radius;


    glPushMatrix();
    glTranslated(v[0][0], v[0][1], v[0][2]);
    glRotated(90, 0, 0, 1);
    glRotated(270, 0, 1, 0);
    DrawOpenDomeQuater(radius, RESOLUTION, RESOLUTION);
    glPopMatrix();

    glPushMatrix();
    glTranslated(v[1][0], v[1][1], v[1][2]);
    glRotated(90, 0, 0, 1);
    DrawOpenDomeQuater(radius, RESOLUTION, RESOLUTION);
    glPopMatrix();

    glPushMatrix();
    glTranslated(v[2][0], v[2][1], v[2][2]);
    DrawOpenDomeQuater(radius, RESOLUTION, RESOLUTION);
    glPopMatrix();

    glPushMatrix();
    glTranslated(v[3][0], v[3][1], v[3][2]);
    glRotated(-90, 0, 1, 0);
    DrawOpenDomeQuater(radius, RESOLUTION, RESOLUTION);
    glPopMatrix();

    glPushMatrix();
    glTranslated(v[4][0], v[4][1], v[4][2]);
    glRotated(180, 0, 0, 1);
    glRotated(-90, 0, 1, 0);
    DrawOpenDomeQuater(radius, RESOLUTION, RESOLUTION);
    glPopMatrix();

    glPushMatrix();
    glTranslated(v[5][0], v[5][1], v[5][2]);
    glRotated(180, 0, 0, 1);
    DrawOpenDomeQuater(radius, RESOLUTION, RESOLUTION);
    glPopMatrix();

    glPushMatrix();
    glTranslated(v[6][0], v[6][1], v[6][2]);
    glRotated(90, 0, 1, 0);
    DrawOpenDomeQuater(radius, RESOLUTION, RESOLUTION);
    glPopMatrix();

    glPushMatrix();
    glTranslated(v[7][0], v[7][1], v[7][2]);
    glRotated(180, 0, 1, 0);
    DrawOpenDomeQuater(radius, RESOLUTION, RESOLUTION);
    glPopMatrix();
}

void DrawRoundedBoxCylinderByParts(double height, double radius, double startDegree, double endDegree, int slides){
    glScaled(radius, height, radius);

    double gap = (endDegree - startDegree) / slides;

    for (int i=0;i<slides;i++){
        double t1 = (startDegree + i * gap) * M_PI / 180.0;
        double t2 = (startDegree + (i + 1) * gap) * M_PI / 180.0;

        double x0 = 1.0 * cos(t1);
        double z0 = 1.0 * sin(t1);

        double x1 = 1.0 * cos(t2);
        double z1 = 1.0 * sin(t2);
#if(USE_VNORMAL == 1)
        //    TODO
#else
        glBegin(GL_QUADS);
        glNormal3f(x0, 0, z0);
        glVertex3f(x0, 0.5, z0);
        glNormal3f(x0, 0, z0);
        glVertex3f(x0, -0.5, z0);
        glNormal3f(x1, 0, z1);
        glVertex3f(x1, -0.5, z1);
        glNormal3f(x1, 0, z1);
        glVertex3f(x1, 0.5, z1);

        glEnd();
#endif

    }
}
void GUI::DrawRoundedBoxCylinder(const Eigen::Vector2d& size, double height, double radius){
    glPushMatrix();
    glTranslated(size[0]/2 - radius, 0, size[1]/2 - radius);
    DrawRoundedBoxCylinderByParts(height, radius, 90, 0, RESOLUTION);
    glPopMatrix();

    glPushMatrix();
    glTranslated(size[0]/2 - radius, 0, -size[1]/2 + radius);
    DrawRoundedBoxCylinderByParts(height, radius, 0, -90, RESOLUTION);
    glPopMatrix();

    glPushMatrix();
    glTranslated(-size[0]/2 + radius, 0, -size[1]/2 + radius);
    DrawRoundedBoxCylinderByParts(height, radius, -90, -180, RESOLUTION);
    glPopMatrix();

    glPushMatrix();
    glTranslated(-size[0]/2 + radius, 0, size[1]/2 - radius);
    DrawRoundedBoxCylinderByParts(height, radius, -180, -270, RESOLUTION);
    glPopMatrix();



}



void GUI::DrawRoundedBoxPlanes(const Eigen::Vector3d& size, double radius)
{
//    glScaled(_size(0), _size(1), _size(2));


    // Code taken from glut/lib/glut_shapes.c
    static GLfloat n[6][3] =
            {
                    {-1.0, 0.0, 0.0},
                    {0.0, 1.0, 0.0},
                    {1.0, 0.0, 0.0},
                    {0.0, -1.0, 0.0},
                    {0.0, 0.0, 1.0},
                    {0.0, 0.0, -1.0}
            };
    static GLfloat vn[8][3] =
            {
                    {-1.0/3.0, -1.0/3.0, -1.0/3.0},
                    {-1.0/3.0, -1.0/3.0, 1.0/3.0},
                    {-1.0/3.0, 1.0/3.0, 1.0/3.0},
                    {-1.0/3.0, 1.0/3.0, -1.0/3.0},
                    {1.0/3.0, -1.0/3.0, -1.0/3.0},
                    {1.0/3.0, -1.0/3.0, 1.0/3.0},
                    {1.0/3.0, 1.0/3.0, 1.0/3.0},
                    {1.0/3.0, 1.0/3.0, -1.0/3.0}
            };
    static GLint faces[6][4] =
            {
                    {0, 1, 2, 3},
                    {3, 2, 6, 7},
                    {7, 6, 5, 4},
                    {4, 5, 1, 0},
                    {5, 6, 2, 1},
                    {7, 4, 0, 3}
            };

    GLfloat faceCenter[6][3] =
            {
                    {-(GLfloat)(size[0] / 2.0), 0.0, 0.0},
                    {0.0, (GLfloat)(size[1] / 2.0), 0.0},
                    {(GLfloat)(size[0] / 2.0), 0.0, 0.0},
                    {0.0, -(GLfloat)(size[1] / 2.0), 0.0},
                    {0.0, 0.0, (GLfloat)(size[2] / 2.0)},
                    {0.0, 0.0, -(GLfloat)(size[2] / 2.0)}
            };

    GLfloat v[8][3];
    GLint i;

    v[0][0] = v[1][0] = v[2][0] = v[3][0] = -size[0] / 2;
    v[4][0] = v[5][0] = v[6][0] = v[7][0] = size[0] / 2;
    v[0][1] = v[1][1] = v[4][1] = v[5][1] = -size[1] / 2;
    v[2][1] = v[3][1] = v[6][1] = v[7][1] = size[1] / 2;
    v[0][2] = v[3][2] = v[4][2] = v[7][2] = -size[2] / 2;
    v[1][2] = v[2][2] = v[5][2] = v[6][2] = size[2] / 2;

#if(USE_VNORMAL == 1)
//    TODO
#else
    for (i = 5; i >= 0; i--) {
        glBegin(GL_QUADS);
        glNormal3fv(&n[i][0]);
        GLfloat _v[3];
        moveToCenter(v[faces[i][0]], faceCenter[i], radius, _v);
        glVertex3fv(&_v[0]);
        moveToCenter(v[faces[i][1]], faceCenter[i], radius, _v);
        glVertex3fv(&_v[0]);
        moveToCenter(v[faces[i][2]], faceCenter[i], radius, _v);
        glVertex3fv(&_v[0]);
        moveToCenter(v[faces[i][3]], faceCenter[i], radius, _v);
        glVertex3fv(&_v[0]);
        glEnd();
    }
#endif
}

void
GUI::
DrawCube(const Eigen::Vector3d& _size)
{
    glScaled(_size(0), _size(1), _size(2));

    // Code taken from glut/lib/glut_shapes.c
    static GLfloat n[6][3] =
    {
        {-1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, -1.0, 0.0},
        {0.0, 0.0, 1.0},
        {0.0, 0.0, -1.0}
    };
    static GLfloat vn[8][3] =
    {
        {-1.0/3.0, -1.0/3.0, -1.0/3.0},
        {-1.0/3.0, -1.0/3.0, 1.0/3.0},
        {-1.0/3.0, 1.0/3.0, 1.0/3.0},
        {-1.0/3.0, 1.0/3.0, -1.0/3.0},
        {1.0/3.0, -1.0/3.0, -1.0/3.0},
        {1.0/3.0, -1.0/3.0, 1.0/3.0},
        {1.0/3.0, 1.0/3.0, 1.0/3.0},
        {1.0/3.0, 1.0/3.0, -1.0/3.0}
    };
    static GLint faces[6][4] =
    {
        {0, 1, 2, 3},
        {3, 2, 6, 7},
        {7, 6, 5, 4},
        {4, 5, 1, 0},
        {5, 6, 2, 1},
        {7, 4, 0, 3}
    };
    GLfloat v[8][3];
    GLint i;
    GLfloat size = 1;

    v[0][0] = v[1][0] = v[2][0] = v[3][0] = -size / 2;
    v[4][0] = v[5][0] = v[6][0] = v[7][0] = size / 2;
    v[0][1] = v[1][1] = v[4][1] = v[5][1] = -size / 2;
    v[2][1] = v[3][1] = v[6][1] = v[7][1] = size / 2;
    v[0][2] = v[3][2] = v[4][2] = v[7][2] = -size / 2;
    v[1][2] = v[2][2] = v[5][2] = v[6][2] = size / 2;
#if(USE_VNORMAL == 1)
    for (i = 5; i >= 0; i--) {
        glBegin(GL_QUADS);
        glNormal3fv(&vn[faces[i][0]][0]);
        glVertex3fv(&v[faces[i][0]][0]);

        glNormal3fv(&vn[faces[i][1]][0]);
        glVertex3fv(&v[faces[i][1]][0]);

        glNormal3fv(&vn[faces[i][2]][0]);
        glVertex3fv(&v[faces[i][2]][0]);

        glNormal3fv(&vn[faces[i][3]][0]);
        glVertex3fv(&v[faces[i][3]][0]);
        glEnd();
    }
#else
    for (i = 5; i >= 0; i--) {
        glBegin(GL_QUADS);
        glNormal3fv(&n[i][0]);
        glVertex3fv(&v[faces[i][0]][0]);
        glVertex3fv(&v[faces[i][1]][0]);
        glVertex3fv(&v[faces[i][2]][0]);
        glVertex3fv(&v[faces[i][3]][0]);
        glEnd();
    }
#endif
}
void
GUI::
DrawTetrahedron(const Eigen::Vector3d& p0,const Eigen::Vector3d& p1,const Eigen::Vector3d& p2,const Eigen::Vector3d& p3,const Eigen::Vector3d& color)
{
	DrawTriangle(p0,p1,p2,color);
	DrawTriangle(p0,p1,p3,color);
	DrawTriangle(p0,p2,p3,color);
	DrawTriangle(p1,p2,p3,color);
}
void
GUI::
DrawTriangle(const Eigen::Vector3d& p0,const Eigen::Vector3d& p1,const Eigen::Vector3d& p2,const Eigen::Vector3d& color)
{
	glColor3f(color[0],color[1],color[2]);
	glBegin(GL_TRIANGLES);
	glVertex3f(p0[0],p0[1],p0[2]);
	glVertex3f(p1[0],p1[1],p1[2]);
	glVertex3f(p2[0],p2[1],p2[2]);
	glEnd();
}
void
GUI::
DrawLine(const Eigen::Vector3d& p0,const Eigen::Vector3d& p1,const Eigen::Vector3d& color)
{
	glColor3f(color[0],color[1],color[2]);
	glBegin(GL_LINES);
    glNormal3f(0.0, 1.0, 0.0);
	glVertex3f(p0[0],p0[1],p0[2]);
	glVertex3f(p1[0],p1[1],p1[2]);
	glEnd();
}
void
GUI::
DrawPoint(const Eigen::Vector3d& p0,const Eigen::Vector3d& color)
{
	glColor3f(color[0],color[1],color[2]);
	glBegin(GL_POINTS);
    glNormal3f(0.0, 1.0, 0.0);
	glVertex3f(p0[0],p0[1],p0[2]);
	glEnd();
}
void
GUI::
DrawArrow3D(const Eigen::Vector3d& _pt, const Eigen::Vector3d& _dir,
            const double _length, const double _thickness,const Eigen::Vector3d& color,
            const double _arrowThickness)
{
    glColor3f(color[0],color[1],color[2]);
    Eigen::Vector3d normDir = _dir;
  normDir.normalize();

  double arrowLength;
  if (_arrowThickness == -1)
    arrowLength = 4*_thickness;
  else
    arrowLength = 2*_arrowThickness;

  // draw the arrow body as a cylinder
  GLUquadricObj *c;
  c = gluNewQuadric();
  gluQuadricDrawStyle(c, GLU_FILL);
  gluQuadricNormals(c, GLU_SMOOTH);

  glPushMatrix();
  glTranslatef(_pt[0], _pt[1], _pt[2]);
  glRotated(acos(normDir[2])*180/M_PI, -normDir[1], normDir[0], 0);
  gluCylinder(c, _thickness, _thickness, _length-arrowLength, 16, 16);

  // draw the arrowhed as a cone
  glPushMatrix();
  glTranslatef(0, 0, _length-arrowLength);
  gluCylinder(c, arrowLength*0.5, 0.0, arrowLength, 10, 10);
  glPopMatrix();

  glPopMatrix();

  gluDeleteQuadric(c);
}

void recursiveRender(const struct aiScene *sc, const struct aiNode* nd) {
    unsigned int i;
    unsigned int n = 0, t;
    aiMatrix4x4 m = nd->mTransformation;

    // update transform
    aiTransposeMatrix4(&m);
    glPushMatrix();
    glMultMatrixf((float*)&m);

    // draw all meshes assigned to this node
    for (; n < nd->mNumMeshes; ++n) {
        const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

        glPushAttrib(GL_POLYGON_BIT | GL_LIGHTING_BIT);  // for applyMaterial()

        if(mesh->mNormals == nullptr) { glDisable(GL_LIGHTING);
        } else {
            glEnable(GL_LIGHTING);
        }

        for (t = 0; t < mesh->mNumFaces; ++t) {
            const struct aiFace* face = &mesh->mFaces[t];
            GLenum face_mode;

            switch(face->mNumIndices) {
                case 1: face_mode = GL_POINTS; break;
                case 2: face_mode = GL_LINES; break;
                case 3: face_mode = GL_TRIANGLES; break;
                default: face_mode = GL_POLYGON; break;
            }

            glBegin(face_mode);

            for (i = 0; i < face->mNumIndices; i++) {
                int index = face->mIndices[i];
                if(mesh->mColors[0] != nullptr)
                    glColor4fv((GLfloat*)&mesh->mColors[0][index]);
                if(mesh->mNormals != nullptr)
                    glNormal3fv(&mesh->mNormals[index].x);
                glVertex3fv(&mesh->mVertices[index].x);
            }

            glEnd();
        }

        glPopAttrib();  // for applyMaterial()
    }

    // draw all children
    for (n = 0; n < nd->mNumChildren; ++n) {
        recursiveRender(sc, nd->mChildren[n]);
    }

    glPopMatrix();
}
// void
// GUI::
// drawCylinder(double _radius, double _height,const Eigen::Vector3d& color, int slices, int stacks)
// {
//   glColor3f(color[0],color[1],color[2]);
//   glPushMatrix();

//   // Graphics assumes Cylinder is centered at CoM
//   // gluCylinder places base at z = 0 and top at z = height
//   glTranslated(0.0, 0.0, -0.5*height);

//   // Code taken from glut/lib/glut_shapes.c
//   QUAD_OBJ_INIT;
//   gluQuadricDrawStyle(quadObj, GLU_FILL);
//   gluQuadricNormals(quadObj, GLU_SMOOTH);
//   //gluQuadricTexture(quadObj, GL_TRUE);

//   // glut/lib/glut_shapes.c
//   gluCylinder(quadObj, _radius, _radius, _height, slices, stacks);
//   glPopMatrix();
//   glPushMatrix();
//   glTranslated(0.0, 0.0, 0.5*height);
//   DrawSphere(radius*2);
//   glPopMatrix();
//   glPushMatrix();
//   glTranslated(0.0, 0.0, -0.5*height);
//   DrawSphere(radius*2);
//   glPopMatrix();
// }
void
GUI::
DrawBezierCurve(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2,
    const Eigen::Vector3d& color)
{
    glColor3f(color[0],color[1],color[2]);
    glBegin(GL_LINE_STRIP);
    for(double s = 0;s<=1.0;s+=0.05)
    {
        Eigen::Vector3d p = 
            p0*(1-s)*(1-s)+
            p1*2*s*(1-s)+
            p2*s*s;

        glVertex3f(p[0],p[1],p[2]);
    }
    glEnd();
}

void
GUI::
DrawMesh(const Eigen::Vector3d& scale, const aiScene* mesh,const Eigen::Vector3d& color)
{
 if (!mesh)
    return;
  glColor3f(color[0],color[1],color[2]);
  glPushMatrix();

  glScaled(scale[0], scale[1], scale[2]);
  recursiveRender(mesh, mesh->mRootNode);

  glPopMatrix();
}


void
GUI::
DrawStringOnScreen(float _x, float _y, const std::string& _s,bool _bigFont,const Eigen::Vector3d& color)
{
    glColor3f(color[0],color[1],color[2]);
	
    // draws text on the screen
    GLint oldMode;
    glGetIntegerv(GL_MATRIX_MODE, &oldMode);
    glMatrixMode(GL_PROJECTION);

    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRasterPos2f(_x, _y);
    unsigned int length = _s.length();
    for (unsigned int c = 0; c < length; c++) {
    if (_bigFont)
      glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, _s.at(c) );
    else
      glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, _s.at(c) );
    }  
    glPopMatrix();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(oldMode);
}

void 
GUI::
DrawGround(int com_x, int com_z, double ground_height){
    static float ground_mat_shininess[] = {128.0};
    static float ground_mat_specular1[]  = {0.0, 0.0, 0.0, 0.};

    static float ground_mat_diffuse1[]   = {1.0, 1.0, 1.0, 0.36};
    static float ground_mat_ambient1[]  = {1.0, 1.0, 1.0, 0.36};
//    static float ground_mat_diffuse1[]   = {0.001, 0.001, 0.001, 0.36};
//    static float ground_mat_ambient1[]  = {0.001, 0.001, 0.001, 0.36};
    static float ground_mat_specular2[]  = {0.0, 0.0, 0.0, 0.};
    static float ground_mat_diffuse2[]   = {1.0, 1.0, 1.0, 0.36};
    static float ground_mat_ambient2[]  = {1.0, 1.0, 1.0, 0.36};

    static float line_mat_specular[] = {0, 0, 0, 0.36};
    static float line_mat_diffuse[] = {0, 0, 0, 0.36};
    static float line_mat_ambient[] = {0, 0, 0, 0.36};


    // Eigen::Vector3d com_root = this->mWorld->getSkeleton("Humanoid")->getRootBodyNode()->getCOM();
    glEnable(GL_LIGHTING);
    double radius = 30;
    double radius_b = radius+2;
    double num_pieces = 1;
    double len = 1.0/num_pieces;
    for(int x=-radius_b; x<=radius_b; x+=1){
        for(int z=-radius_b; z<=radius_b; z+=1){
            if((x+com_x+z+com_z)%2 == 0){
                // glColor3f(0.7, 0.7, 0.7);
                glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, ground_mat_shininess);
                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  ground_mat_specular1);
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   ground_mat_diffuse1);
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,   ground_mat_ambient1);
                glBegin(GL_QUADS);
                glNormal3f(0.0, 1.0, 0.0);
                glVertex3f(x+com_x,ground_height,z+com_z);
                glVertex3f(x+com_x,ground_height,z+com_z+1);
                glVertex3f(x+com_x+1,ground_height,z+com_z+1);
                glVertex3f(x+com_x+1,ground_height,z+com_z);
                glEnd();
            }
            else{
                // glColor3f(0.9, 0.9, 0.9);
                glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, ground_mat_shininess);
                glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  ground_mat_specular2);
                glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   ground_mat_diffuse2);
                glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,   ground_mat_ambient2);
                glBegin(GL_QUADS);
                glNormal3f(0.0, 1.0, 0.0);
                glVertex3f(x+com_x,ground_height,z+com_z);
                glVertex3f(x+com_x,ground_height,z+com_z+1);
                glVertex3f(x+com_x+1,ground_height,z+com_z+1);
                glVertex3f(x+com_x+1,ground_height,z+com_z);
                glEnd();
            }
        }
    }

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  line_mat_specular);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   line_mat_diffuse);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,   line_mat_ambient);
    for(int x=-radius_b; x<=radius_b; x+=1){
        glLineWidth(3.0);
        glBegin(GL_LINE_STRIP);
        glNormal3f(0.0, 1.0, 0.0);
        glVertex3f(x+com_x, 0, -radius_b+com_z);
        glVertex3f(x+com_x, 0, +radius_b+com_z);
        glEnd();


        glEnd();

        glBegin(GL_LINE_STRIP);
        glNormal3f(0.0, 1.0, 0.0);
        glVertex3f(-radius_b+com_x,0,x+com_z);
        glVertex3f(+radius_b+com_x,0,x+com_z);
        glEnd();
    }
    double wall_height = 150;
    static float ground_mat_specular3[]  = {0.0, 0.0, 0.0, 0.0};
    static float ground_mat_diffuse3[]   = {0.0, 0.0, 0.0, 1.0};
    static float ground_mat_ambient3[]  = {0.0, 0.0, 0.0, 1.0};
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  ground_mat_specular3);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   ground_mat_diffuse3);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT,   ground_mat_ambient3);
    glBegin(GL_QUADS);
        glNormal3f(0.0, 0.0, 1.0);
        glVertex3f(com_x-radius_b,ground_height-wall_height,com_z-radius_b);
        glNormal3f(0.0, 0.0, 1.0);
        glVertex3f(com_x-radius_b,ground_height+wall_height,com_z-radius_b);
        glNormal3f(0.0, 0.0, 1.0);
        glVertex3f(com_x+radius_b+1,ground_height+wall_height,com_z-radius_b);
        glNormal3f(0.0, 0.0, 1.0);
        glVertex3f(com_x+radius_b+1,ground_height-wall_height,com_z-radius_b);

        glNormal3f(0.0, 0.0, -1.0);
        glVertex3f(com_x-radius_b,ground_height-wall_height,com_z+radius_b+1);
        glNormal3f(0.0, 0.0, -1.0);
        glVertex3f(com_x-radius_b,ground_height+wall_height,com_z+radius_b+1);
        glNormal3f(0.0, 0.0, -1.0);
        glVertex3f(com_x+radius_b+1,ground_height+wall_height,com_z+radius_b+1);
        glNormal3f(0.0, 0.0, -1.0);
        glVertex3f(com_x+radius_b+1,ground_height-wall_height,com_z+radius_b+1);

        glNormal3f(1.0, 0.0, 0.0);
        glVertex3f(com_x-radius_b,ground_height-wall_height,com_z+radius_b+1);
        glNormal3f(1.0, 0.0, 0.0);
        glVertex3f(com_x-radius_b,ground_height+wall_height,com_z+radius_b+1);
        glNormal3f(1.0, 0.0, 0.0);
        glVertex3f(com_x-radius_b,ground_height+wall_height,com_z-radius_b);
        glNormal3f(1.0, 0.0, 0.0);
        glVertex3f(com_x-radius_b,ground_height-wall_height,com_z-radius_b);

        glNormal3f(-1.0, 0.0, 0.0);
        glVertex3f(com_x+radius_b+1,ground_height-wall_height,com_z+radius_b+1);
        glNormal3f(-1.0, 0.0, 0.0);
        glVertex3f(com_x+radius_b+1,ground_height+wall_height,com_z+radius_b+1);
        glNormal3f(-1.0, 0.0, 0.0);
        glVertex3f(com_x+radius_b+1,ground_height+wall_height,com_z-radius_b);
        glNormal3f(-1.0, 0.0, 0.0);
        glVertex3f(com_x+radius_b+1,ground_height-wall_height,com_z-radius_b);
    glEnd();
}