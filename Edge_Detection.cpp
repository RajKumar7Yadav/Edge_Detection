#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#define Pi 3.14

using namespace cv;
using namespace std;

typedef vector<double>Array;
typedef vector<Array>Matrix;

Mat RGB2Gray(Mat input)
{
	Mat gray(input.rows,input.cols,CV_8UC1);
	for (int i = 0; i < gray.rows; i++)
	{
		for (int j = 0; j < gray.cols; j++)
		{
			Vec3b pixel = input.at<Vec3b>(i, j);
			gray.at<uchar>(i, j) = 0.3 * pixel[2] + 0.59 * pixel[1] + 0.11 * pixel[0];
		}
	}
	return gray;
}

Mat Gaussian_Blur(Mat gray_img)
{
	Mat gblur_img(gray_img.rows, gray_img.cols, CV_8UC1);
	for (int i = 1; i < gray_img.rows - 1; i++)
	{
		for (int j = 1; j < gray_img.cols - 1; j++)
		{
			gblur_img.at<uchar>(i, j) = (gray_img.at<uchar>(i-1,j-1) * 1 +gray_img.at<uchar>(i-1,j) * 2 + gray_img.at<uchar>(i-1,j+1) * 1
				+ gray_img.at<uchar>(i,j-1) * 2 + gray_img.at<uchar>(i,j) * 4 + gray_img.at<uchar>(i,j+1)*2
				+ gray_img.at<uchar>(i+1,j-1) * 1 + gray_img.at<uchar>(i+1,j) * 2 + gray_img.at<uchar>(i+1,j+1) * 1) / 16;
		}
	}
	return gblur_img;
}

int xgradient(Mat gblur, int i, int j)
{
	int gx = gblur.at<uchar>(i - 1, j - 1)*(-1) + gblur.at<uchar>(i - 1, j)*0 + gblur.at<uchar>(i - 1, j + 1)*1
		+ gblur.at<uchar>(i, j - 1)*(-2) + gblur.at<uchar>(i, j)*0 + gblur.at<uchar>(i, j + 1)*(2)
		+ gblur.at<uchar>(i + 1, j - 1)*(-1) + gblur.at<uchar>(i + 1, j)*0 + gblur.at<uchar>(i + 1, j + 1)*1;
	return gx;
}

int ygradient(Mat gblur, int i, int j)
{
	int gy = gblur.at<uchar>(i - 1, j - 1) * (-1) + gblur.at<uchar>(i - 1, j) *(-2) + gblur.at<uchar>(i - 1, j + 1) *(-1)
		+ gblur.at<uchar>(i, j - 1) * (0) + gblur.at<uchar>(i, j) * 0 + gblur.at<uchar>(i, j + 1) * (0)
		+ gblur.at<uchar>(i + 1, j - 1) *(1) + gblur.at<uchar>(i + 1, j) * 2 + gblur.at<uchar>(i + 1, j + 1) * 1;
	return gy;
}

double getThetha(int gy, int gx)
{
	double thetha;
	if (gx == 0)thetha = 0;
	else
	{
		thetha = (atan(gy / gx) * 180) / 3.1415;
	}
	return thetha;
}
/*
Mat non_maximum_suppression(Mat Sobel_img, Matrix thetha)
{
	// Normalizing the thetha value

}
*/
void findthetha(Matrix thetha)
{
	for (int i = 0; i < thetha.size(); i++)
	{
		for (int j = 0; j < thetha[0].size(); j++)
		{
			if (thetha[i][j]>180)
			{
				cout << thetha[i][j] << endl;
			}
		}
	}
}

Mat non_maximum_suppression(Mat Sobel_img, Matrix thetha)
{
	// 1. Normalizing the thetha value
	for (int i = 0; i < thetha.size(); i++)
	{
		for (int j = 0; j < thetha[0].size(); j++)
		{
			if ((thetha[i][j] > -22.5 && thetha[i][j] < 22.5) && (thetha[i][j] > -157.5 && thetha[i][j] < 157.5))
				thetha[i][j] = 0;
			if ((thetha[i][j] > 22.5 && thetha[i][j] < 67.5) && (thetha[i][j] < -112.5 && thetha[i][j] > -157.5))
				thetha[i][j] = 45;
			if ((thetha[i][j] > 67.5 && thetha[i][j] < 112.5) && (thetha[i][j] < -67.5 && thetha[i][j] > -112.5))
				thetha[i][j] = 90;
			if ((thetha[i][j] > 112.5 && thetha[i][j] < 157.5) && (thetha[i][j] < -22.5 && thetha[i][j] > -67.5))
				thetha[i][j] = 135;
		}
	}
	// Now we apply non-maximum suppression
	Mat Sobel_img_supp=Sobel_img.clone();
	for (int i = 1; i < Sobel_img.rows-1; i++)
	{
		for (int j = 1; j < Sobel_img.cols-1; j++)
		{
			if (thetha[i][j] == 0)
			{
				if (Sobel_img.at<uchar>(i, j) > Sobel_img.at<uchar>(i, j - 1) && Sobel_img.at<uchar>(i, j) > Sobel_img.at<uchar>(i, j + 1))
					Sobel_img_supp.at<uchar>(i, j) = Sobel_img.at<uchar>(i, j);
				else
					Sobel_img_supp.at<uchar>(i, j) = 0;
			}
			if (thetha[i][j] == 45)
			{
				if (Sobel_img.at<uchar>(i, j) > Sobel_img.at<uchar>(i+1,j+1) && Sobel_img.at<uchar>(i,j) > Sobel_img.at<uchar>(i-1,j+1))
					Sobel_img_supp.at<uchar>(i, j) = Sobel_img.at<uchar>(i, j);
				else
					Sobel_img_supp.at<uchar>(i, j) = 0;
			}
			if (thetha[i][j] == 90)
			{
				if (Sobel_img.at<uchar>(i, j) > Sobel_img.at<uchar>(i-1,j) && Sobel_img.at<uchar>(i,j) > Sobel_img.at<uchar>(i+1,j))
					Sobel_img_supp.at<uchar>(i, j) = Sobel_img.at<uchar>(i, j);
				else
					Sobel_img_supp.at<uchar>(i, j) = 0;
			}
			if (thetha[i][j] == 135)
			{
				if (Sobel_img.at<uchar>(i, j) > Sobel_img.at<uchar>(i-1,j-1) && Sobel_img.at<uchar>(i,j) > Sobel_img.at<uchar>(i+1,j-1))
					Sobel_img_supp.at<uchar>(i, j) = Sobel_img.at<uchar>(i, j);
				else
					Sobel_img_supp.at<uchar>(i, j) = 0;
			}
		}
	}
	return Sobel_img_supp;
}

Mat Sobelfilter(Mat gblur)
{
	int gx, gy, G;
	Matrix thetha(gblur.rows, Array(gblur.cols));
	Matrix Sobel_img(gblur.rows, Array(gblur.cols));
	Mat Sobel_img1 = gblur.clone();
	for (int i = 1; i < gblur.rows - 1; i++)
	{
		for (int j = 1; j < gblur.cols - 1; j++)
		{
			gx = xgradient(gblur,i,j);
			gy = ygradient(gblur,i,j);
			G = sqrt(gx * gx + gy * gy);//abs(gx) + abs(gy);
			G = G > 255 ? 255 : G;
			G = G < 0 ? 0 : G;
			Sobel_img1.at<uchar>(i, j) = G;
			Sobel_img[i][j] = G;
			thetha[i][j] = getThetha(gy, gx);
		}
	}
	// check the negative values of thetha
	//findthetha(thetha);
	//Step 4: Non maximum suppression
	Mat Sobel_img_supp = non_maximum_suppression(Sobel_img1,thetha);
	return Sobel_img_supp;
}

bool bob_analysis(Mat sobel_img, int i, int j,int maxthresh)
{
	if (sobel_img.at<uchar>(i - 1, j - 1) > maxthresh || sobel_img.at<uchar>(i - 1, j) > maxthresh || sobel_img.at<uchar>(i - 1, j + 1) > maxthresh
		|| sobel_img.at<uchar>(i, j - 1) > maxthresh || sobel_img.at<uchar>(i, j + 1) > maxthresh ||
		sobel_img.at<uchar>(i + 1, j - 1) > maxthresh || sobel_img.at<uchar>(i + 1, j) > maxthresh || sobel_img.at<uchar>(i + 1, j + 1) > maxthresh)
		return true;
	else
		return false;
}

Mat double_threshold(Mat sobel_img)
{
	Mat edge_img = sobel_img.clone();
	int maxthresh = 200;
	int minthresh = 170;
	for (int i = 1; i < sobel_img.rows - 1; i++)
	{
		for (int j = 1; j < sobel_img.cols - 1; j++)
		{
			if (sobel_img.at<uchar>(i, j) < minthresh)
				edge_img.at<uchar>(i, j) = 0;
			if (sobel_img.at<uchar>(i, j) < minthresh && sobel_img.at<uchar>(i, j) > maxthresh)
			{
				bool strength = bob_analysis(sobel_img, i, j, maxthresh);
				if (strength ==false)
					edge_img.at<uchar>(i,j) = 0;

			}
		}
	}
	return edge_img;
}

int main()
{
	Mat input = imread("lane.jpg");
	// Step 1: Convert RGB image to Gray image
	Mat gray = RGB2Gray(input);
	//imshow("Gray", gray);

	// Step 2: Apply Gaussian Blur 
	Mat gblur = Gaussian_Blur(gray);
	//imshow("Gaussian Blur",gblur);

	// Step 3: Apply Sobel Filter
	Mat sobel_img = Sobelfilter(gblur);
	imshow("Sobel Image", sobel_img);
	
	// Step 5: Applying double threshold
	Mat edge_img = double_threshold(sobel_img);
	imshow("Edge Image",edge_img);
	waitKey(0);
}