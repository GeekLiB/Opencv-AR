#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <windows.h>
#include <iostream>
#include <pthread.h>
#include <GL/glut.h>
#include <math.h>

using namespace cv;
using namespace std;

VideoCapture cap(0);
static int resolution_width = 640;
static int resolution_height = 480;
static int num_track = 4;

double camD[9] = {618.526381968738, 0, 310.8963715614199,
	0, 619.4548980786033, 248.6374860176724,
	0, 0, 1};
double distCoeffD[5] = {0.09367405350511771, -0.08731677320554751, 0.002823563134787144, -1.246739177460954e-005, -0.0469061739387372};
Mat camera_matrix = Mat(3,3,CV_64FC1,camD);
Mat distortion_coefficients = Mat(5,1,CV_64FC1,distCoeffD);

vector<Point3f> objP;
Mat objPM;
vector<double> rv(3), tv(3);
Mat rvecm(rv),tvecm(tv); 


Mat gray, prevGray, image, frame;
vector<Point2f> points[2];
vector<KeyPoint> keypoints;
vector<KeyPoint> initPoints;
vector<Point2f> recoveryPoints;
vector<Point2f> projectedPoints;
vector<Point2f> goodfeatures;
bool initflag = true;
bool needToGetgf = false;
bool needtomap = false;
bool needtokeeptime = false;

const int MAX_COUNT = 500;
size_t trackingpoints = 0;

Mat initdescriptors;

string msg;
int baseLine;
Size textSize;

DWORD t1,t2;
int framenum = 0;

TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 30, 0.01);
Size subPixWinSize(10,10),winSize(21,21);


static void help()
{
	// print a welcome message, and the OpenCV version	
	cout << "\n<Feature Tracking and Synchronous Scene Generation with a Single Camera>\n"
		"\nUser: cc.\n"
		"\nUsing OpenCV version " << CV_VERSION << endl;
	cout << "\nHot keys: \n"
		"    Tracking: \n"
		"\tESC - quit the program\n"
		"\tg - get the good corners\n"
		"\tc - delete all the points\n"
		"\tm - choose four points to track and update camera pose then\n"
		"\t\t(To add a tracking point, please click that point)\n" 
		"    Generation: \n"
		"\tESC - quit the program\n"
		"\te - change the view mode\n"<< endl;

}

void on_mouse(int event,int x,int y,int flag, void *param)
{
	if(event==CV_EVENT_LBUTTONDOWN)
	{
		if(needtomap && points[1].size()<num_track)
		{
			for(size_t i = 0;i<goodfeatures.size();i++)
			{
				if(abs(goodfeatures[i].x-x)+abs(goodfeatures[i].y-y)<5)
				{
					points[1].push_back(goodfeatures[i]);
					initPoints.push_back(keypoints[i]);
					trackingpoints++;
					break;
				}
			}
			Mat temp;
			image.copyTo(temp);
			for(size_t i = 0; i < points[1].size(); i++ )
			{
				circle( temp, points[1][i], 3, Scalar(0,0,255), -1, 8);
			}

			msg = format( "Resolution: %d * %d.  Corner number: %d.  Tracked points: %d",(int)cap.get(CV_CAP_PROP_FRAME_WIDTH),(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT),goodfeatures.size(),trackingpoints);
			baseLine = 0;
			textSize = getTextSize(msg, 1, 1, 1, &baseLine);
			Point textOrigin(temp.cols - textSize.width - 20, temp.rows - 2*baseLine - 10);
			putText(temp,msg,textOrigin,1,1,Scalar(0,0,255));

			imshow("Tracking", temp);
		}	
	}
}

void init3DP()
{
	// 	objP.push_back(Point3f(-5,-5,0));    //三维坐标的单位是毫米
	// 	objP.push_back(Point3f(0,-5,0));
	// 	objP.push_back(Point3f(5,-5,0));
	// 	objP.push_back(Point3f(5,0,0));
	// 	objP.push_back(Point3f(5,5,0));    
	// 	objP.push_back(Point3f(0,5,0));
	// 	objP.push_back(Point3f(-5,5,0));
	// 	objP.push_back(Point3f(-5,0,0));
	// 	objP.push_back(Point3f(0,0,0));
	if(!objP.empty())
		objP.clear();
	objP.push_back(Point3f(-0.5,-0.5,0));    //三维坐标的单位是毫米
	objP.push_back(Point3f(0.5,-0.5,0)); 
	objP.push_back(Point3f(0.5,0.5,0)); 
	objP.push_back(Point3f(-0.5,0.5,0));
	objPM.setTo(0);
	Mat(objP).convertTo(objPM,CV_32F);
}

bool init()
{
	cap.set(CV_CAP_PROP_FRAME_WIDTH,resolution_width);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,resolution_height);

	if( !cap.isOpened() )
	{
		cout << "Could not initialize capturing...\n";
		return false;
	}

	namedWindow( "Tracking", 1 );
	cvSetMouseCallback( "Tracking",on_mouse,NULL );

	init3DP();
	return true;
}

void getPlanarSurface(vector<Point2f>& imgP){

	solvePnP(objPM, Mat(imgP), camera_matrix, distortion_coefficients, rvecm, tvecm);

	projectedPoints.clear();
	projectPoints(objPM, rvecm, tvecm, camera_matrix, distortion_coefficients, projectedPoints);

	for(unsigned int i = 0; i < projectedPoints.size(); ++i)
	{
		circle( image, projectedPoints[i], 3, Scalar(255,0,0), -1, 8);
	}
}

bool attemptoRecoveryTracking(vector<Point2f> missingPoints)
{
	DWORD t_s,t_e,t = 0;
	ORB orb;
	BFMatcher bfm;
	Mat descriptors;
	Mat tempF,tempI,tempG;
	Mat top,bottom,left,right;
	vector<KeyPoint> newKP;
	vector<DMatch> matches;
	int locationflag = 0;

	t_s = GetTickCount();
	while(t<5000)
	{

		cap >> tempF;			   
		if( tempF.empty() )
			break;
		tempF.copyTo(tempI);

		if(missingPoints[0].x<30 && missingPoints[3].x<30)
		{
			left = tempI(Range(0,480),Range(0,200));
			cvtColor(left, tempG, COLOR_BGR2GRAY);
			locationflag = 1;
		}
		else if(missingPoints[1].x>610 && missingPoints[2].x>610)
		{
			right = tempI(Range(0,480),Range(440,640));
			cvtColor(right, tempG, COLOR_BGR2GRAY);
			locationflag = 2;
		}
		else if(missingPoints[0].y<30 && missingPoints[1].y<30)
		{
			top = tempI(Range(0,200),Range(0,640));
			cvtColor(top, tempG, COLOR_BGR2GRAY);
			locationflag = 3;
		}
		else if(missingPoints[2].y>450 && missingPoints[3].y>450)
		{
			bottom = tempI(Range(280,480),Range(0,640));
			cvtColor(bottom, tempG, COLOR_BGR2GRAY);
			locationflag = 4;
		}
		else{
			cvtColor(tempI, tempG, COLOR_BGR2GRAY);
		}
		orb.detect(tempG,newKP);
		orb.compute(tempG,newKP,descriptors);
		bfm.match(initdescriptors,descriptors,matches);

		recoveryPoints.clear();
		if(matches.size() == num_track)		
		{
			for(size_t i = 0; i < num_track; i++ )
			{
				Point t = newKP[matches[i].trainIdx].pt;
				if(locationflag == 2)
				{
					t.x += 440;
				}
				if(locationflag == 4)
				{
					t.y += 280;
				}
				recoveryPoints.push_back(t);
			}
			return true;
		}
		imshow("Tracking",tempI);
		waitKey(1);
		t_e = GetTickCount();
		t = t_e - t_s;
	}
	return false;
}

bool attemptoRecoveryTracking2(int mpi)
{
	vector<Point3f> newobjP;
	for(int i = 0;i<num_track;i++)
	{
		if(i == mpi)
			continue;
		newobjP.push_back(objP.at(i));
	}
	switch (mpi)
	{
	case 0:
		newobjP.push_back(Point3f(1.5,-0.5,0));
		break;
	case 1:
		newobjP.push_back(Point3f(0.5,1.5,0));
		break;
	case 2:
		newobjP.push_back(Point3f(-1.5,0.5,0));
		break;
	case 3:
		newobjP.push_back(Point3f(-0.5,-1.5,0));
		break;
	default:
		break;
	}
	objPM.setTo(0);
	Mat(newobjP).convertTo(objPM,CV_32F);
	points[1].clear();
	projectPoints(objPM, rvecm, tvecm, camera_matrix, distortion_coefficients, points[1]);
	objP = newobjP;
	return true;
}

int tracking_update()
{
	ORB orb;
	for(;;)
	{
		if( needtomap && goodfeatures.size()>0 )
		{
			needToGetgf = false;
			if(trackingpoints<num_track)
			{
				char c = waitKey(2);
				if(c == 'c')
				{
					points[0].clear();
					points[1].clear();
					trackingpoints = 0;
					goodfeatures.clear();
					needToGetgf = false;
					needtomap = false;
					initflag = true;
					initPoints.clear();
					init3DP();
				}
				if(c == 27)
					break;
				continue;
			}
			if(initflag && initPoints.size()==num_track)
			{
				ORB orb;
				cvtColor(image, gray, COLOR_BGR2GRAY);
				orb.compute(gray,initPoints,initdescriptors);
				initflag = false;
			}
			cap >> frame;
			if( frame.empty() )
				break;
			frame.copyTo(image);
			cvtColor(image, gray, COLOR_BGR2GRAY);

			if(!points[0].empty())
			{
				vector<uchar> status;
				vector<float> err;
				if(prevGray.empty())
					gray.copyTo(prevGray);
				calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err);
				size_t i,k;
				int missingPointIndex = 0;
				for(i = k = 0; i < points[1].size(); i++ )
				{
					if( !status[i] )
					{
						missingPointIndex = i;
						continue;
					}
					points[1][k++] = points[1][i];
					circle( image, points[1][i], 3, Scalar(0,0,255), -1, 8);
				}
				if(k == num_track)		
				{
					getPlanarSurface(points[0]);
				}
				else
				{
					if(attemptoRecoveryTracking2(missingPointIndex))
					{}
					else
					{
						needtomap = false;
						needtokeeptime = false;
						trackingpoints = 0;
						points[0].clear();
						points[1].clear();
						goodfeatures.clear();
						initflag = true;
						initPoints.clear();
					}
				}
			}	
			framenum++;
		}
		else
		{
			cap >> frame;
			if( frame.empty() )
				break;

			frame.copyTo(image);
			if(needToGetgf)
			{
				cvtColor(image, gray, COLOR_BGR2GRAY);

				// automatic initialization
//				goodFeaturesToTrack(gray, goodfeatures, MAX_COUNT, 0.01, 10);
				orb.detect(gray, keypoints);
				goodfeatures.clear();
				for( size_t i = 0; i < keypoints.size(); i++ ) {
					goodfeatures.push_back(keypoints[i].pt);
				}
				cornerSubPix(gray, goodfeatures, subPixWinSize, Size(-1,-1), termcrit);
				for(size_t i = 0; i < goodfeatures.size(); i++ )
				{
					circle( image, goodfeatures[i], 3, Scalar(0,255,0), -1, 8);
				}
			}
		}

		msg = format( "Resolution: %d * %d.  Corner number: %d.  Tracked points: %d",
			(int)cap.get(CV_CAP_PROP_FRAME_WIDTH),(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT),goodfeatures.size(),trackingpoints);
		baseLine = 0;
		textSize = getTextSize(msg, 1, 1, 1, &baseLine);
		Point textOrigin(image.cols - textSize.width - 20, image.rows - 2*baseLine - 10);
		putText(image,msg,textOrigin,1,1,Scalar(0,0,255));

		imshow("Tracking", image);

		char c = (char)waitKey(5);
		if( c == 27 )
		{
			cap.release();
			return 0;
		}

		switch( c )
		{
		case 'g':
			needToGetgf = true;
			needtomap = false;
			break;
		case 'c':
			points[0].clear();
			points[1].clear();
			trackingpoints = 0;
			goodfeatures.clear();
			needToGetgf = false;
			needtomap = false;
			initflag = true;
			initPoints.clear();
			init3DP();
			break;
		case 'm':
			trackingpoints = 0;
			needtomap = true;
			break;
		case 't':
			needtokeeptime = true;
			framenum=0;
			t1 = GetTickCount();
			break;
		}

		std::swap(points[1], points[0]);
		cv::swap(prevGray, gray);
		framenum++;
	}
	if(needtokeeptime)
	{
		t2 = GetTickCount();
		cout<<endl<<"fps:"<<framenum/((t2-t1)*1.0/1000)<<"\n";
	}
	return 0;
}

void* tracking_thread(void *p)
{
	help();

	if(!init())
		return NULL;

	tracking_update();

	return NULL;
}


const GLfloat PI = 3.14;

/// record the state of mouse
GLboolean mousedown = GL_FALSE;
GLboolean drawmap = GL_TRUE;
/// when a mouse-key is pressed, record current mouse position 
static GLint mousex = 0, mousey = 0;

static GLfloat center[3] = {0.0f, 0.0f, 0.0f}; /// center position
static GLfloat eye[3]; /// eye's position

static GLfloat yrotate = PI/2; /// angle between y-axis and look direction
static GLfloat xrotate = PI/2; /// angle between x-axis and look direction
static GLfloat celength = 15.0f;/// lenght between center and eye

// GLfloat light_ambient[] = {1.0, 1.0, 1.0, 1.0};  /* Red diffuse light. */
// GLfloat light_diffuse[] = {1.0, 0.0, 0.0, 1.0};  /* Red diffuse light. */
// GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};  /* Infinite light location. */

Mat texttmp(resolution_width,resolution_height,CV_8UC3);
double fovx,fovy,focalLength,aspectRatio,z_near = 1.0, z_far = 1000.0; 
Point2d principalPt;

void CalEyePostion()
{
	if(yrotate > PI/1.1) yrotate = PI/1.1;   /// 限制看得方向
	if(yrotate < 0.5)  yrotate = 0.5;
	if(xrotate > PI/1.1)   xrotate = PI/1.1;
	if(xrotate < 0.5)   xrotate = 0.5;
	if(celength > 50)  celength = 50;     ///  缩放距离限制
	if(celength < 5)   celength = 5;
	/// 下面利用球坐标系计算 eye 的位置，
	eye[0] = center[0] + celength * sin(yrotate) * cos(xrotate);  
	eye[2] = center[2] + celength * sin(yrotate) * sin(xrotate);
	eye[1] = center[1] + celength * cos(yrotate);
}

void keyboard(unsigned char c,int x,int y)
{
	switch (c)
	{
	case 'e':
		drawmap = !drawmap;
		break;
	case 27:
		exit(0);
	default:
		break;
	}
	glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
	if(state == GLUT_DOWN)
	{
		mousedown = GL_TRUE;
	}
	mousex = x, mousey = y;
}

void motion(int x,int y)
{
	if(mousedown == GL_TRUE && !drawmap)
	{       /// 所除以的数字是调整旋转速度的，随便设置，达到自己想要速度即可
		xrotate += (x - mousex) / 80.0f;     
		yrotate -= (y - mousey) / 120.0f;
	}
	mousex = x, mousey = y;
	CalEyePostion();
	glutPostRedisplay();
}

void draw_object()
{
	glColor3f(1.0,0.0,0.0);						// 右为X正方向，上为Y正方向，屏幕外为Z正方向
	glBegin(GL_QUADS);                          //  绘制顶面(逆时针绘制为正面)
	glVertex3f(-0.5f, 0.5f, 0.5f);                 
	glVertex3f(-0.5f, 1.5f, 0.5f);                  
	glVertex3f(0.5f, 1.5f, 0.5f);                   
	glVertex3f(0.5f, 0.5f, 0.5f);                    
	glEnd();

	glColor3f(0.0,1.0,0.0);
	glBegin(GL_QUADS);                          //  绘制前面(逆时针绘制为正面)
	glVertex3f(-0.5f, -0.5f, 0.5f);                 
	glVertex3f(-0.5f, -0.5f, -0.5f);                  
	glVertex3f(0.5f, -0.5f, -0.5f);                   
	glVertex3f(0.5f, -0.5f, 0.5f);                    
	glEnd();  

	glColor3f(0.0,0.0,1.0);
	glBegin(GL_QUADS);                          //  绘制左面(逆时针绘制为正面)
	glVertex3f(-0.5f, 0.5f, 0.5f);                 
	glVertex3f(-0.5f, 0.5f, -0.5f);                  
	glVertex3f(-0.5f, -0.5f, -0.5f);                   
	glVertex3f(-0.5f, -0.5f, 0.5f);                    
	glEnd();  

	glColor3f(1.0,1.0,0.0);
	glBegin(GL_QUADS);                          //  绘制右面(逆时针绘制为正面)
	glVertex3f(0.5f, -0.5f, 0.5f);                 
	glVertex3f(0.5f, -0.5f, -0.5f);                  
	glVertex3f(0.5f, 0.5f, -0.5f);                   
	glVertex3f(0.5f, 0.5f, 0.5f);                    
	glEnd();     

	glColor3f(1.0,0.0,1.0);
	glBegin(GL_QUADS);                          //  绘制后面(逆时针绘制为正面)
	glVertex3f(0.5f, 0.5f, 0.5f);                 
	glVertex3f(0.5f, 0.5f, -0.5f);                  
	glVertex3f(-0.5f, 0.5f, -0.5f);                   
	glVertex3f(-0.5f, 0.5f, 0.5f);                    
	glEnd();    

	glColor3f(0.0,1.0,1.0);
	glBegin(GL_QUADS);                          //  绘制底面(逆时针绘制为正面)
	glVertex3f(0.5f, 0.5f, -0.5f);                 
	glVertex3f(0.5f, -0.5f, -0.5f);                  
	glVertex3f(-0.5f, -0.5f, -0.5f);                   
	glVertex3f(-0.5f, 0.5f, -0.5f);                    
	glEnd(); 
}

void draw_map()
{
	if(frame.data!=NULL)
	{
		cvtColor(frame, texttmp, CV_BGR2RGB);
		flip(texttmp,texttmp,0);
		glEnable(GL_TEXTURE_2D);
		glTexImage2D(GL_TEXTURE_2D, 0, 3, resolution_width, resolution_height	, 0, GL_RGB, GL_UNSIGNED_BYTE, texttmp.data);

		glPushMatrix();

		glScaled(1.0/resolution_width, 1.0/resolution_height, 1.0);
		glScaled(1.55, 1.55, 1);
		glTranslated(-resolution_width/2, -resolution_height/2, 0.0);
		glBegin(GL_QUADS);
		glTexCoord2i(0, 0); glVertex2i(0,	0);
		glTexCoord2i(1, 0); glVertex2i(resolution_width, 0);
		glTexCoord2i(1, 1); glVertex2i(resolution_width, resolution_height);
		glTexCoord2i(0, 1); glVertex2i(0,	resolution_height);
		glEnd();

		glPopMatrix();
		glDisable(GL_TEXTURE_2D);
	}

	if(needtomap && trackingpoints == num_track)
	{
		/* Use depth buffering for hidden surface elimination. */
		glEnable(GL_DEPTH_TEST);
		Mat rotM(3,3,CV_64FC1);
		Rodrigues(rvecm,rotM);

		glPushMatrix();
		double model_view_matrix[16]={
			rotM.at<double>(0,0),-rotM.at<double>(1,0),-rotM.at<double>(2,0),0,
			rotM.at<double>(0,1),-rotM.at<double>(1,1),-rotM.at<double>(2,1),0,
			rotM.at<double>(0,2),-rotM.at<double>(1,2),-rotM.at<double>(2,2),0,
			tv[0],-tv[1],-tv[2],1
		};
		glLoadMatrixd(model_view_matrix);

		glRotated(180.0,1.0,0.0,0.0);

		draw_object();        

		/* Use depth buffering for hidden surface elimination. */
		glDisable(GL_DEPTH_TEST);
		glPopMatrix();
	}
}

void draw_camera()
{
	glColor3f(1.0,1.0,1.0);

	glPushMatrix();
	glScalef(0.25,0.25,0.25);
	glBegin(GL_LINES);
	for(float i = -5;i<5.1;i+=0.5)
	{
		glVertex3f(-5,i,0);
		glVertex3f(5,i,0);
		glVertex3f(i,-5,0);
		glVertex3f(i,5,0);
	}
	glEnd();

	glBegin(GL_LINES);
	for(int i = -50;i<51;i+=5)
	{
		glVertex3i(-50,i,0);
		glVertex3i(50,i,0);
		glVertex3i(i,-50,0);
		glVertex3i(i,50,0);
	}
	glEnd();

	glColor3f(1.0,0.0,0.0);
	glBegin(GL_LINES);
	glVertex3i(-50,0,0);
	glVertex3i(100,0,0);
	glEnd();

	glColor3f(0.0,1.0,0.0);
	glBegin(GL_LINES);
	glVertex3i(0,-50,0);
	glVertex3i(0,100,0);
	glEnd();

	glColor3f(0.0,0.0,1.0);
	glBegin(GL_LINES);
	glVertex3i(0,0,-10);
	glVertex3i(0,0,50);
	glEnd();

	glEnable(GL_DEPTH_TEST);
	draw_object();
	glDisable(GL_DEPTH_TEST);

	if(needtomap && trackingpoints == num_track)
	{
		glPushMatrix();
		Mat tempR;

		Mat rotM(3,3,CV_64FC1);
		Rodrigues(rvecm,rotM);

		rotM.copyTo(tempR);
		tempR = tempR.t();
		vector<double> tempv(3);
		Mat tempT(tempv);
		tvecm.copyTo(tempT);
		tempT = tempR * tempT;

		glTranslated(-tempv[0],tempv[1],tempv[2]);
		glutSolidCube(0.5);
		glPopMatrix();
	}

	glPopMatrix();
}

void display(void)
{
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLoadIdentity();  //加载单位矩阵

	if(drawmap)
	{
		gluLookAt(0,0,2,
			0,0,0,
			0,1,0);
		draw_map();
	}
	else
	{
		CalEyePostion();
		gluLookAt(eye[0], eye[1], eye[2],
			center[0], center[1], center[2],
			0, 1, 0);
		draw_camera();
	}

	waitKey(15);
	glutSwapBuffers();
}

void idle() { glutPostRedisplay(); }

void reshape (int w, int h)
{
	glViewport (0, 0, (GLsizei) w, (GLsizei) h); 

	/* Setup the view of the cube. */
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity ();

	gluPerspective( /* field of view in degree */ fovy,
		/* aspect ratio */ 1.0/aspectRatio,
		/* Z near */ z_near, /* Z far */ z_far);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	CalEyePostion();
	gluLookAt(eye[0], eye[1], eye[2],
		center[0], center[1], center[2],
		0, 1, 0);
}

void initGL()
{
	glClearColor (0.0, 0.0, 0.0, 0.0); //背景黑色
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);

	// 	/* Enable a single OpenGL light. */
	// 	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	// 	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	// 	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	// 	glEnable(GL_LIGHT0);
	// 	glEnable(GL_LIGHTING);

	calibrationMatrixValues(camera_matrix, Size(resolution_width,resolution_height), 0.0, 0.0, fovx, fovy, focalLength, principalPt, aspectRatio);
}

int main(int argc, char** argv)           
{
	//	glutInit(&argc, argv);
	glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize (resolution_width, resolution_height); 
	glutInitWindowPosition (100, 300);
	glutCreateWindow ("Generation");
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	glutMouseFunc(mouse);
	glutDisplayFunc(display); 
	glutIdleFunc(idle);
	glutReshapeFunc(reshape);
	initGL();

	pthread_t pid; 
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_PROCESS);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	pthread_create(&pid, &attr, tracking_thread, NULL);

	glutMainLoop();

	return 0;
}
