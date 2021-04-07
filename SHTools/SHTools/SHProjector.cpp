
#include "SH.h"

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <eigen3/Eigen/Eigen>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static Eigen::MatrixXf inv_M1(3,3);
static Eigen::MatrixXf inv_M2(5,5);
static Eigen::MatrixXf inv_M3(7,7);
static Eigen::MatrixXf inv_M4(9,9);
static Eigen::MatrixXf sample_N1(3,3);
static Eigen::MatrixXf sample_N2(3,5);
static Eigen::MatrixXf sample_N3(3,7);
static Eigen::MatrixXf sample_N4(3,9);

void Rotate_SH_L1(float ,SHCoefficients& );
void Rotate_SH_L2(float ,SHCoefficients& );
void Rotate_SH_L3(float ,SHCoefficients& );
void Rotate_SH_L4(float ,SHCoefficients& );
void Rotate_SH_L5(float ,SHCoefficients& );

class Image
{
public:
	Image(int w, int h, int d, float* dat) : width(w), depth(d), data(dat) { assert(w == h); }

	float& at(int i, int j, int k)
	{
		return data[(i * width + j) * depth + k];
	}

	float at(int i, int j, int k) const
	{
		return data[(i * width + j) * depth + k];
	}

	int width, depth;

private:
	float* data;
};


float sinc(float x)
{               /* Supporting sinc function */
	if (fabs(x) < 1.0e-4) return 1.0;
	else return(sin(x) / x);
}


// Refer to the paper: 2002 An Efficient Representation for Irradiance Environment Maps
bool sphereMapping(int i, int j, int width, float& x, float& y, float& z, float& domega)
{
	float u, v, r, theta, phi;

	v = (width / 2.0 - i) / (width / 2.0);		// v ranges from -1 to 1
	u = (j - width / 2.0) / (width / 2.0);		// u ranges from -1 to 1
	r = sqrt(u * u + v * v);

	// Consider only circle with r < 1
	if (r > 1.0)
		return false;

	theta = PI * r;
	phi = atan2(v, u);

	// Cartesian components
	x = sin(theta) * cos(phi);
	y = sin(theta) * sin(phi);
	z = cos(theta);

	// Computation of the solid angle.  This follows from some elementary calculus converting sin(theta) d theta d phi into
    // coordinates in terms of r.  This calculation should be redone if the form of the input changes
	domega = (2 * PI / width) * (2 * PI / width) * sinc(theta);

	return true;
}


SHColor project(FILTER_TYPE filterType, const Image& img)
{
	SHColor shColor;

	for (int i = 0; i < img.width; i++)
	{
		for (int j = 0; j < img.width; j++)
		{
			// We now find the cartesian components for the point (i,j)
			float x, y, z, domega;
            //std::cout<<i<<" "<<j<<":";
			if (sphereMapping(i, j, img.width, x, y, z, domega))
			{
				float color[3] = { img.at(i, j, 0), img.at(i, j, 1), img.at(i,j,2) };

				// Update Integration
				//std::cout<<i<<" "<<j<<":";
//				if(x>0&&y>0){
//                    std::cout<<x<<" "<<y<<" "<<z<<std::endl;
//				}

				SHCoefficients shDirection;
				projectOnSH(x, y, z, shDirection);
				// order 3
//				float* pSH = shDirection.getPtr();
//				for(int i=0;i<9;i++){
//				    std::cout<<pSH[0]<<" ";
//				}
//				std::cout<<std::endl;
//                Rotate_SH_L1(45,shDirection);
//                pSH = shDirection.getPtr();
//                for(int i=0;i<9;i++){
//                    std::cout<<pSH[0]<<" ";
//                }
//                std::cout<<std::endl;
//                Rotate_SH_L2(45,shDirection);
//                pSH = shDirection.getPtr();
//                for(int i=0;i<9;i++){
//                    std::cout<<pSH[0]<<" ";
//                }
//                std::cout<<std::endl;
//                Rotate_SH_L3(45,shDirection);
//                pSH = shDirection.getPtr();
//                for(int i=0;i<9;i++){
//                    std::cout<<pSH[0]<<" ";
//                }
//                std::cout<<std::endl;
				switch (filterType)
				{
				case FILTER_GAUSSIAN:
					gauss(SH_ORDER, shDirection);
					break;
				case FILTER_HANNING:
					hanning(SH_ORDER, shDirection);
					break;
				case FILTER_LANCZOS:
					lanczos(SH_ORDER, shDirection);
					break;
				default:
					break;
				}

				for (int c = 0; c < 3; ++c)
				{
					for (int k = 0; k < SH_COUNT; ++k)
					{
						shColor[c][k] += shDirection[k] * color[c] * domega;
					}
				}
			}
		}
	}
	return shColor;
}



Eigen::Matrix3f get_rotation_matrix(float rotation_angle)
{
    Eigen::Matrix3f model = Eigen::Matrix3f::Identity();
    float a = rotation_angle / 180 * PI;
    model << cos(a), -sin(a), 0,
            sin(a), cos(a), 0,
            0, 0, 1;
//    model << 1, 0, 0,
//             0, cos(a), -sin(a),
//             0, sin(a), cos(a);
    return model;
}

void get_inverse_matrix_l1_n1(){
    const static float temp0 = sqrt((4*PI)/3.0f);
    sample_N1 << 1.f ,0, 0,
                 0, 1.f, 0,
                 0, 0, 1.f;

    inv_M1 << 0,     0,     temp0,
              temp0, 0,     0,
              0,     temp0, 0;

}
void get_inverse_matrix_l2_n2(){
    const static float tempK = 1 / sqrt(2);
    sample_N2 << 1, 0, tempK, tempK, 0,
            0, 0, tempK, 0,     tempK,
            0, 1, 0,     tempK, tempK;

    const static float temp0 = 0.91529123286551084;
    const static float temp1 = 1.83058246573102168;
    const static float temp2 = 1.5853309190550713;
    inv_M2 << 0,     temp0, 0,      -temp0,  temp1,
            temp0, 0,     temp2,  -temp0,  temp0,
            temp1, 0,     0,      0,       0,
            0,     0,     0,      temp1,   0,
            0,     temp1, 0,      0,       0;
}
void get_inverse_matrix_l3_n3(){
    Eigen::Vector3f n0,n1,n2,n3,n4,n5,n6;
    n0 << 1.f, 0, 0;
    n1 << 0, 1.f, 0;
    n2 << 0.3f, 0, 1.f;
    n2 = n2.normalized();
    n3 << 0, 1.f, 1.f;
    n3 = n3.normalized();
    n4 << 1.f, 0, 1.f;
    n4 = n4.normalized();
    n5 << 1.f, 1.f, 0.78f;
    n5 = n5.normalized();
    n6 << 1.f, 1.f, 1.f;
    n6 = n6.normalized();
    sample_N3 << n0, n1, n2 , n3, n4, n5, n6;
    inv_M3 <<   0.707711955885399,  0.643852929494021,  -0.913652206352009, -0.093033334712756,   0.328680372803511,  -1.131667680791894,   1.949384763080401 ,
            -1.114187338255984,  0.643852929494021,  -0.749554866243252, -0.093033334712757,   0.164583032694754,  -0.232204002745663,   0.127485468939019 ,
            2.296023687102124,                  0,  -2.964153834214758,                  0,   2.964153834214758,  -3.749390980495911,   2.296023687102124 ,
            2.392306681179504, -1.099424142052695,  -3.088454645076318, -2.129025696294232,   3.766408103751610,  -5.313883353254694,   2.917447172170129 ,
            1.878707739441422, -1.099424142052695,  -2.425401262415870, -2.129025696294233,   3.103354721091161,  -2.518204820606409,   2.403848230432046 ,
            13.656934981397061, -4.181565269348606, -17.631027247729438, -8.097566324633245,  14.325209638780166, -20.210898801851609,  11.096259672385109 ,
            -13.139185354460187,  5.820633765367933,  16.962615353518899,  7.790578559853934, -13.782124974734103,  19.444681101542464, -10.675588100498899 ;

}
void get_inverse_matrix_l4_n4(){
    Eigen::Vector3f n0,n1,n2,n3,n4,n5,n6,n7,n8;
    n0 << 1.f ,0, 0;
    n1 << 0, 1.f, 0;
    n2 << 0.3, 0, 1.f;
    n2 = n2.normalized();
    n3 << 0, 1.f, 1.f;
    n3 = n3.normalized();
    n4 << 1.f, 0, 1.f;
    n4 = n4.normalized();
    n5 << 1.f, 0.54f, 0.78f;
    n5 = n5.normalized();
    n6 << 1, 1, 0.78;
    n6 = n6.normalized();
    n7 << 0.31, 1, 0.78;
    n7 = n7.normalized();
    n8 << 1.f, 1.f, 1.f;
    n8 = n8.normalized();
    sample_N4 << n0, n1, n2, n3, n4, n5, n6, n7, n8;
    inv_M4 <<   -1.948206991589258, 1.912687049138671,    -0.763091021186035, -0.286837642392582, -0.341264679278342, 0.594477634079894,  -1.056887279361603, 0.578857155270682,  0.971984464556520,
            2.171192074917378,  -0.142084581369102, -1.577618721617938, 0.828536347413562,  -0.705532540822805, 0.382031320127708,  1.056887279361603,  -2.513802449733083, 1.156701984383617,
            2.053952330860290,  -0.094158653118148, -0.750956907863241, -1.098731135021785, -0.335838138831051, 1.931188736063331,  0,                  -1.051043414216722, 0.170301019159901,
            3.993132334888566,  1.179414191911931,  -4.808985771815311, 1.266884703225481,  -3.095952538204609, 2.811562290853012,  0,                  -4.022967497037739, 1.569934476060706,
            -1.543780567538975, 1.894449743774703,  -2.499709102566265, -0.207318037527907, -2.063212615945576, 1.725864595116423,  0,                  -0.365404044003703, 1.046239752465574,
            3.435134010827782,  -2.932684025967419, 4.231264528651311,  -2.972023260715974, 1.892279023369589,  -1.718456688280952, 0,                  2.458880397035034,  -0.959560600640598,
            3.689266412234284,  1.985158283498190,  -7.403078714786565, -3.123392326177335, -3.310757449808909, 3.006635497533013,  0,                  -4.302091019418769, 1.678860447048080,
            -0.367659806642012, -3.222124483746851, 4.648868038376401,  -3.265346293642776, 2.079036990447149,  -1.888059306949047, 0,                  2.701558933638689,  -1.054264174928627,
            -4.515212732000947, 3.220651333447782,  0.208527587656698,  6.066568738154828,  -0.970215938306426, 0.881093140952614,  0,                  -1.260725782049042, 0.491989276959057;
}



void Rotate_SH_L1(float angle,SHCoefficients& sh){
    Eigen::Matrix3f R = get_rotation_matrix(angle);
    Eigen::Vector3f n0,n1,n2,ori_sh,rot_sh;
    n0 << sample_N1(0,0),sample_N1(1,0),sample_N1(2,0);
    n1 << sample_N1(0,1),sample_N1(1,1),sample_N1(2,1);
    n2 << sample_N1(0,2),sample_N1(1,2),sample_N1(2,2);
    // rotation the sample vectors
    n0 = R * n0;
    n1 = R * n1;
    n2 = R * n2;
    float* pSH = sh.getPtr();
    SHCoefficients sh_n0;
    SHCoefficients sh_n1;
    SHCoefficients sh_n2;
    // calculate sh vectors of after-rotation sample vectors
    projectOnSH(n0.x(), n0.y(), n0.z(),sh_n0);
    projectOnSH(n1.x(), n1.y(), n1.z(),sh_n1);
    projectOnSH(n2.x(), n2.y(), n2.z(),sh_n2);
    Eigen::Vector3f pa0,pa1,pa2;
    pa0 << sh_n0[1], sh_n0[2],sh_n0[3];
    pa1 << sh_n1[1], sh_n1[2],sh_n1[3];
    pa2 << sh_n2[1], sh_n2[2],sh_n2[3];
    Eigen::Matrix3f S, Mr;
    S << pa0, pa1, pa2;
    Mr = S * inv_M1;
    ori_sh << pSH[1], pSH[2],pSH[3];
    rot_sh  = Mr * ori_sh;
//    std::cout<<"before rotation l1:"<<std::endl;
//    std::cout<<ori_sh<<std::endl;
//    std::cout<<"after rotation l1:"<<std::endl;
//    std::cout<<rot_sh<<std::endl;

    pSH[1] = rot_sh(0);
    pSH[2] = rot_sh(1);
    pSH[3] = rot_sh(2);
}

void Rotate_SH_L2(float angle,SHCoefficients& sh){
    Eigen::Matrix3f R = get_rotation_matrix(angle);
    Eigen::Vector3f n0,n1,n2,n3, n4;
    n0 << sample_N2(0,0),sample_N2(1,0),sample_N2(2,0);
    n1 << sample_N2(0,1),sample_N2(1,1),sample_N2(2,1);
    n2 << sample_N2(0,2),sample_N2(1,2),sample_N2(2,2);
    n3 << sample_N2(0,3),sample_N2(1,3),sample_N2(2,3);
    n4 << sample_N2(0,4),sample_N2(1,4),sample_N2(2,4);

    // rotation the sample vectors
    n0 = R * n0;
    n1 = R * n1;
    n2 = R * n2;
    n3 = R * n3;
    n4 = R * n4;

    float* pSH = sh.getPtr();
    SHCoefficients sh_n0;
    SHCoefficients sh_n1;
    SHCoefficients sh_n2;
    SHCoefficients sh_n3;
    SHCoefficients sh_n4;
    // calculate sh vectors of after-rotation sample vectors
    projectOnSH(n0.x(), n0.y(), n0.z(),sh_n0);
    projectOnSH(n1.x(), n1.y(), n1.z(),sh_n1);
    projectOnSH(n2.x(), n2.y(), n2.z(),sh_n2);
    projectOnSH(n3.x(), n3.y(), n3.z(),sh_n3);
    projectOnSH(n4.x(), n4.y(), n4.z(),sh_n4);
    Eigen::VectorXf pa0(5),pa1(5),pa2(5),pa3(5),pa4(5);
    pa0 << sh_n0[4], sh_n0[5],sh_n0[6],sh_n0[7],sh_n0[8];
    pa1 << sh_n1[4], sh_n1[5],sh_n1[6],sh_n1[7],sh_n1[8];
    pa2 << sh_n2[4], sh_n2[5],sh_n2[6],sh_n2[7],sh_n2[8];
    pa3 << sh_n3[4], sh_n3[5],sh_n3[6],sh_n3[7],sh_n3[8];
    pa4 << sh_n4[4], sh_n4[5],sh_n4[6],sh_n4[7],sh_n4[8];
    Eigen::MatrixXf S(5,5), Mr(5,5);
    S << pa0, pa1, pa2, pa3, pa4;
    Mr = S * inv_M2;
    Eigen::VectorXf ori_sh(5),rot_sh(5);
    ori_sh << pSH[4], pSH[5],pSH[6],pSH[7],pSH[8];
    rot_sh  = Mr * ori_sh;
    pSH[4] = rot_sh(0);
    pSH[5] = rot_sh(1);
    pSH[6] = rot_sh(2);
    pSH[7] = rot_sh(3);
    pSH[8] = rot_sh(4);
//    std::cout<<"before rotation l2:"<<std::endl;
//    std::cout<<ori_sh<<std::endl;
//    std::cout<<"after rotation l2:"<<std::endl;
//    std::cout<<rot_sh<<std::endl;
}

void Rotate_SH_L3(float angle,SHCoefficients& sh){
    Eigen::Matrix3f R = get_rotation_matrix(angle);
    Eigen::Vector3f n0,n1,n2,n3, n4,n5,n6;
    n0 << sample_N3(0,0),sample_N3(1,0),sample_N3(2,0);
    n1 << sample_N3(0,1),sample_N3(1,1),sample_N3(2,1);
    n2 << sample_N3(0,2),sample_N3(1,2),sample_N3(2,2);
    n3 << sample_N3(0,3),sample_N3(1,3),sample_N3(2,3);
    n4 << sample_N3(0,4),sample_N3(1,4),sample_N3(2,4);
    n5 << sample_N3(0,5),sample_N3(1,5),sample_N3(2,5);
    n6 << sample_N3(0,6),sample_N3(1,6),sample_N3(2,6);

    // rotation the sample vectors
    n0 = R * n0;
    n1 = R * n1;
    n2 = R * n2;
    n3 = R * n3;
    n4 = R * n4;
    n5 = R * n5;
    n6 = R * n6;

    float* pSH = sh.getPtr();
    SHCoefficients sh_n0;
    SHCoefficients sh_n1;
    SHCoefficients sh_n2;
    SHCoefficients sh_n3;
    SHCoefficients sh_n4;
    SHCoefficients sh_n5;
    SHCoefficients sh_n6;
    // calculate sh vectors of after-rotation sample vectors
    projectOnSH(n0.x(), n0.y(), n0.z(),sh_n0);
    projectOnSH(n1.x(), n1.y(), n1.z(),sh_n1);
    projectOnSH(n2.x(), n2.y(), n2.z(),sh_n2);
    projectOnSH(n3.x(), n3.y(), n3.z(),sh_n3);
    projectOnSH(n4.x(), n4.y(), n4.z(),sh_n4);
    projectOnSH(n5.x(), n5.y(), n5.z(),sh_n5);
    projectOnSH(n6.x(), n6.y(), n6.z(),sh_n6);

    Eigen::VectorXf pa0(7),pa1(7),pa2(7),pa3(7),pa4(7),pa5(7),pa6(7);
    pa0 << sh_n0[9], sh_n0[10],sh_n0[11],sh_n0[12],sh_n0[13],sh_n0[14],sh_n0[15];
    pa1 << sh_n1[9], sh_n1[10],sh_n1[11],sh_n1[12],sh_n1[13],sh_n1[14],sh_n1[15];
    pa2 << sh_n2[9], sh_n2[10],sh_n2[11],sh_n2[12],sh_n2[13],sh_n2[14],sh_n2[15];
    pa3 << sh_n3[9], sh_n3[10],sh_n3[11],sh_n3[12],sh_n3[13],sh_n3[14],sh_n3[15];
    pa4 << sh_n4[9], sh_n4[10],sh_n4[11],sh_n4[12],sh_n4[13],sh_n4[14],sh_n4[15];
    pa5 << sh_n5[9], sh_n5[10],sh_n5[11],sh_n5[12],sh_n5[13],sh_n5[14],sh_n5[15];
    pa6 << sh_n6[9], sh_n6[10],sh_n6[11],sh_n6[12],sh_n6[13],sh_n6[14],sh_n6[15];
    Eigen::MatrixXf S(7,7), Mr(7,7);
    S << pa0, pa1, pa2, pa3, pa4, pa5, pa6;
    Mr = S * inv_M3;
    Eigen::VectorXf ori_sh(7),rot_sh(7);
    ori_sh << pSH[9], pSH[10],pSH[11],pSH[12],pSH[13],pSH[14],pSH[15];
    rot_sh  = Mr * ori_sh;
    pSH[9] = rot_sh(0);
    pSH[10] = rot_sh(1);
    pSH[11] = rot_sh(2);
    pSH[12] = rot_sh(3);
    pSH[13] = rot_sh(4);
    pSH[14] = rot_sh(5);
    pSH[15] = rot_sh(6);
}

void Rotate_SH_L4(float angle,SHCoefficients& sh){
    Eigen::Matrix3f R = get_rotation_matrix(angle);
    Eigen::Vector3f n0,n1,n2,n3, n4,n5,n6,n7,n8;
    n0 << sample_N4(0,0),sample_N4(1,0),sample_N4(2,0);
    n1 << sample_N4(0,1),sample_N4(1,1),sample_N4(2,1);
    n2 << sample_N4(0,2),sample_N4(1,2),sample_N4(2,2);
    n3 << sample_N4(0,3),sample_N4(1,3),sample_N4(2,3);
    n4 << sample_N4(0,4),sample_N4(1,4),sample_N4(2,4);
    n5 << sample_N4(0,5),sample_N4(1,5),sample_N4(2,5);
    n6 << sample_N4(0,6),sample_N4(1,6),sample_N4(2,6);
    n7 << sample_N4(0,7),sample_N4(1,7),sample_N4(2,7);
    n8 << sample_N4(0,8),sample_N4(1,8),sample_N4(2,8);

    // rotation the sample vectors
    n0 = R * n0;
    n1 = R * n1;
    n2 = R * n2;
    n3 = R * n3;
    n4 = R * n4;
    n5 = R * n5;
    n6 = R * n6;
    n7 = R * n7;
    n8 = R * n8;
    float* pSH = sh.getPtr();
    SHCoefficients sh_n0;
    SHCoefficients sh_n1;
    SHCoefficients sh_n2;
    SHCoefficients sh_n3;
    SHCoefficients sh_n4;
    SHCoefficients sh_n5;
    SHCoefficients sh_n6;
    SHCoefficients sh_n7;
    SHCoefficients sh_n8;
    // calculate sh vectors of after-rotation sample vectors
    projectOnSH(n0.x(), n0.y(), n0.z(),sh_n0);
    projectOnSH(n1.x(), n1.y(), n1.z(),sh_n1);
    projectOnSH(n2.x(), n2.y(), n2.z(),sh_n2);
    projectOnSH(n3.x(), n3.y(), n3.z(),sh_n3);
    projectOnSH(n4.x(), n4.y(), n4.z(),sh_n4);
    projectOnSH(n5.x(), n5.y(), n5.z(),sh_n5);
    projectOnSH(n6.x(), n6.y(), n6.z(),sh_n6);
    projectOnSH(n7.x(), n7.y(), n7.z(),sh_n7);
    projectOnSH(n8.x(), n8.y(), n8.z(),sh_n8);

    Eigen::VectorXf pa0(9),pa1(9),pa2(9),pa3(9),pa4(9),pa5(9),pa6(9),pa7(9),pa8(9);
    pa0 << sh_n0[16], sh_n0[17],sh_n0[18],sh_n0[19],sh_n0[20],sh_n0[21],sh_n0[22],sh_n0[23],sh_n0[24];
    pa1 << sh_n1[16], sh_n1[17],sh_n1[18],sh_n1[19],sh_n1[20],sh_n1[21],sh_n1[22],sh_n1[23],sh_n1[24];
    pa2 << sh_n2[16], sh_n2[17],sh_n2[18],sh_n2[19],sh_n2[20],sh_n2[21],sh_n2[22],sh_n2[23],sh_n2[24];
    pa3 << sh_n3[16], sh_n3[17],sh_n3[18],sh_n3[19],sh_n3[20],sh_n3[21],sh_n3[22],sh_n3[23],sh_n3[24];
    pa4 << sh_n4[16], sh_n4[17],sh_n4[18],sh_n4[19],sh_n4[20],sh_n4[21],sh_n4[22],sh_n4[23],sh_n4[24];
    pa5 << sh_n5[16], sh_n5[17],sh_n5[18],sh_n5[19],sh_n5[20],sh_n5[21],sh_n5[22],sh_n5[23],sh_n5[24];
    pa6 << sh_n6[16], sh_n6[17],sh_n6[18],sh_n6[19],sh_n6[20],sh_n6[21],sh_n6[22],sh_n6[23],sh_n6[24];
    pa7 << sh_n7[16], sh_n7[17],sh_n7[18],sh_n7[19],sh_n7[20],sh_n7[21],sh_n7[22],sh_n7[23],sh_n7[24];
    pa8 << sh_n8[16], sh_n8[17],sh_n8[18],sh_n8[19],sh_n8[20],sh_n8[21],sh_n8[22],sh_n8[23],sh_n8[24];
    Eigen::MatrixXf S(9, 9), Mr(9, 9);
    S << pa0, pa1, pa2, pa3, pa4, pa5, pa6, pa7, pa8;
    Mr = S * inv_M4;
    Eigen::VectorXf ori_sh(9),rot_sh(9);
    ori_sh << pSH[16], pSH[17],pSH[18],pSH[19],pSH[20],pSH[21],pSH[22],pSH[23],pSH[24];
    rot_sh  = Mr * ori_sh;
    pSH[16] = rot_sh(0);
    pSH[17] = rot_sh(1);
    pSH[18] = rot_sh(2);
    pSH[19] = rot_sh(3);
    pSH[20] = rot_sh(4);
    pSH[21] = rot_sh(5);
    pSH[22] = rot_sh(6);
    pSH[23] = rot_sh(7);
    pSH[24] = rot_sh(8);
}
void reconstructFromSH(float x, float y, float z, const SHColor& shColor, float color[3])
{
    SHCoefficients shDirection;
    projectOnSH(x, y, z, shDirection);
    Rotate_SH_L1(45,shDirection);
    Rotate_SH_L2(45,shDirection);
    Rotate_SH_L3(45,shDirection);
    Rotate_SH_L4(45,shDirection);
    color[0] = color[1] = color[2] = 0.0f;
    for (int c = 0; c < 3; ++c)
    {
        for (int i = 0; i < SH_COUNT; ++i)
        {
            color[c] += shColor[c][i] * shDirection[i];
        }
    }
}

void reconstruct(Image& img, const SHColor& shColor)
{
	for (int i = 0; i < img.width; i++)
	{
		for (int j = 0; j < img.width; j++)
		{
			// We now find the cartesian components for the point (i,j)
			float x, y, z, domega;
			if (sphereMapping(i, j, img.width, x, y, z, domega))
			{
				float color[3];
				reconstructFromSH(x, y, z, shColor, color);

				for (int c = 0; c < 3; ++c)
					img.at(i, j, c) = color[c];
			}
		}
	}
}

void printUsageAndExit(const char* argv0)
{
	std::cerr << "Usage  : " << argv0 << " [options] <input>\n";
	std::cerr << "<input>              .hdr probe image\n";
	std::cerr << "Options: --help      Print this usage message\n";
	std::cerr << "         -h          Hanning  filter\n";
	std::cerr << "         -l          Lanczos  filter\n";
	std::cerr << "         -g          Gaussian filter\n";
	exit(0);
}

int main(int argc, char** argv)
{
    get_inverse_matrix_l1_n1();
    get_inverse_matrix_l2_n2();
    get_inverse_matrix_l3_n3();
    get_inverse_matrix_l4_n4();
	FILTER_TYPE filterType = FILTER_DISABLE;
	std::string infile;

	for (int i = 1; i < argc; ++i)
	{
		const std::string arg = argv[i];
		if (arg == "--help")
		{
			printUsageAndExit(argv[0]);
		}
		else if (arg == "-l")
		{
			filterType = FILTER_LANCZOS;
		}
		else if (arg == "-h")
		{
			filterType = FILTER_HANNING;
		}
		else if (arg == "-g")
		{
			filterType = FILTER_GAUSSIAN;
		}
		else
		{
			infile = arg;
		}
	}

	if (infile.empty())
	{
		std::cerr << "need input file argument" << std::endl;
		printUsageAndExit(argv[0]);
	}

	std::string outfile = infile.substr(0, infile.size() - 4) + "-" + std::to_string(SH_ORDER);
	switch (filterType)
	{
	case FILTER_DISABLE:
		break;
	case FILTER_GAUSSIAN:
		outfile += "-g";
		break;
	case FILTER_HANNING:
		outfile += "-h";
		break;
	case FILTER_LANCZOS:
		outfile += "l";
		break;
	default:
		break;
	}
	outfile += ".hdr";

	int w, h, d;

	// Read .HDR image
	float* data = stbi_loadf(infile.c_str(), &w, &h, &d, 0);
	if (data == nullptr)
	{
		std::cerr << "input file argument is incorrect" << std::endl;
		printUsageAndExit(argv[0]);
	}

	Image img(w, h, d, data);

	// Project on SH
	SHColor shColor = project(filterType, img);

	// Log
	for (int i = 0; i < SH_COUNT; ++i)
	{
		printf("m_params.env_radiance[%d] = make_float3(%9.6f, %9.6f, %9.6f);\n", i, shColor[0][i], shColor[1][i], shColor[2][i]);
	}

	// Reconstruct
	float* newData = new float[w * h * d];
	Image newImg(w, h, d, newData);
	reconstruct(newImg, shColor);
	stbi_write_hdr(outfile.c_str(), w, h, d, newData);

	stbi_image_free(data);
	delete newData;
	return 0;
}
