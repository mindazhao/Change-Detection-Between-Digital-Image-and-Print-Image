#include"iostream"
#include<windows.h>
#include<fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/stitcher.hpp"
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>
#include "callback.h"
#include <hash_set>   
#include <ctime> 
#include<random>
#include "Vikey.h"
#include<math.h>
#define max_my(a, b)  (((a) > (b)) ? (a) : (b))
#define     NO_OBJECT       0  
#define     MIN(x, y)       (((x) < (y)) ? (x) : (y))  
#define     ELEM(img, r, c) (CV_IMAGE_ELEM(img, unsigned char, r, c))  
#define     ONETWO(L, r, c, col) (L[(r) * (col) + c])  

using namespace std;
using namespace cv;
int num_sift = 5000;
vector<double>gray_diff;
vector<size_t>order;

int myrandom(int i) { return std::rand() % i; }
template < typename T>
vector< size_t>  sort_indexes(const vector< T>  & v) {

	// initialize original index locations
	vector< size_t>  idx(v.size());
	for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] >  v[i2]; });

	return idx;
}

class myRectangle{
public:
	int x;
	int y;
	int maxX;
	int maxY;

	myRectangle() :x(0), y(0), maxX(0), maxY(0){}

	myRectangle(int a, int b, int c, int d)
	{
		x = min(a, c);
		y = min(b, d);
		maxX = max(a, c);
		maxY = max(b, d);
	}
};

class myLine{
public:
	Point p1;
	Point p2;

	myLine(Point p3, Point p4)
	{
		if ((abs(p4.x - p3.x)) < 50){
			if (p4.y > p3.y){
				p1 = p3;
				p2 = p4;
			}
			else{
				p1 = p4;
				p2 = p3;
			}
		}
		else{
			if (p4.x > p3.x){
				p1 = p3;
				p2 = p4;
			}
			else{
				p1 = p4;
				p2 = p3;
			}
		}
	}
};


double calcaulte_var(vector<int>v1, vector<int>v2, int l)
{
	vector<double>v;
	double sum = 0;
	double max = 0;
	for (int i = 0; i < v1.size(); i++)
	{
		v.push_back(fabs(double(v1[i] - v2[i])));
		double zhong = (v1[i] + v2[i]) / 2;
		sum += v[i];//*log(abs(zhong-125)+1);
		if (v[i] * log(abs(zhong - 125) + 1)>max)
			max = v[i] * log(abs(zhong - 125) + 1);
	}

	double mean = sum / v.size();
	double sum_var = 0;
	for (int i = 0; i < v.size(); i++)
		sum_var += pow(v[i] - mean, 2);
	return (sum) / v.size();
	//return max;
}
bool inRects(vector<myRectangle> r, int x, int y){
	if (r.size() == 0){
		return false;
	}
	for (int k = 0; k < r.size(); k++){
		if (x >= r[k].x && x <= r[k].maxX && y >= r[k].y && y <= r[k].maxY ||
			x <= r[k].x && x >= r[k].maxX && y <= r[k].y && y >= r[k].maxY){
			return true;
		}

	}
	return false;
}

bool inRect(myRectangle r, Point p){
	if (p.x >= r.x && p.x <= r.maxX && p.y >= r.y && p.y <= r.maxY ||
		p.x <= r.x && p.x >= r.maxX && p.y <= r.y && p.y >= r.maxY){
		return true;
	}
	return false;
}


bool canMerge(myRectangle r1, myRectangle r2){
	if (inRect(r1, Point(r2.x, r2.y)) ||
		inRect(r1, Point(r2.x, r2.maxY)) ||
		inRect(r1, Point(r2.maxX, r2.y)) ||
		inRect(r1, Point(r2.maxX, r2.maxY)) ||
		inRect(r2, Point(r1.x, r1.y)) ||
		inRect(r2, Point(r1.x, r1.maxY)) ||
		inRect(r2, Point(r1.maxX, r1.y)) ||
		inRect(r2, Point(r1.maxX, r1.maxY))){
		if (r1.maxX - r1.x > 300 && r2.maxX - r2.x > 300 || r1.maxY - r1.y > 300 && r2.maxY - r2.y > 300){
			return true;
		}
	}
	return false;
}

vector<myRectangle> mixRects(vector<myRectangle> r){
	vector<myRectangle> results;
	hash_set<int> hs;
	set<int> s;
	for (int i = 0; i < r.size(); i++){
		if (s.find(i) == s.end()){
			for (int j = i; j < r.size(); j++){
				if (canMerge(r[i], r[j])){
					int x = min(r[i].x, r[j].x);
					int y = min(r[i].y, r[j].y);
					int maxY = max(r[i].maxY, r[j].maxY);
					int maxX = max(r[i].maxX, r[j].maxX);
					myRectangle temp = myRectangle(x, y, maxX, maxY);
					r[i] = temp;
					r[j] = temp;
					s.insert(j);
				}
			}
		}
		results.push_back(r[i]);
	}
	return results;
}

bool totalInRect(myRectangle r1, myRectangle r2){
	if (r1.x <= r2.x && r1.y <= r2.y && r1.maxX >= r2.maxX && r1.maxY >= r2.maxY){
		return true;
	}
	return false;
}

vector<myRectangle> findFinalRects(vector<myRectangle> r){
	vector<bool> mix;
	for (int i = 0; i < r.size(); i++)
	{
		mix.push_back(true);
	}
	for (int i = 0; i < r.size(); i++)
	{
		for (int j = 0; j < r.size(); j++){
			if (i != j && mix[i] && mix[j]){
				if (totalInRect(r[i], r[j]))
				{
					mix[j] = false;
				}
			}
		}
	}
	vector<myRectangle> result;
	for (int i = 0; i < r.size(); i++)
	{
		if (mix[i]){
			result.push_back(r[i]);
		}
	}
	return result;
}

int calculate_min(vector<double>v)
{
	double min = INT_MAX;
	int index = -1;
	for (int i = 0; i < v.size(); i++)
	{
		if (min>v[i])
		{
			index = i;
			min = v[i];
		}
	}
	return index;
}

double getThreshVal_Otsu_8u(const cv::Mat& _src)
{
	cv::Size size = _src.size();
	if (_src.isContinuous())
	{
		size.width *= size.height;
		size.height = 1;
	}
	const int N = 256;
	int i, j, h[N] = { 0 };
	for (i = 0; i < size.height; i++)
	{
		const uchar* src = _src.data + _src.step*i;
		for (j = 0; j <= size.width - 4; j += 4)
		{
			int v0 = src[j], v1 = src[j + 1];
			h[v0]++; h[v1]++;
			v0 = src[j + 2]; v1 = src[j + 3];
			h[v0]++; h[v1]++;
		}
		for (; j < size.width; j++)
			h[src[j]]++;
	}

	double mu = 0, scale = 1. / (size.width*size.height);
	for (i = 0; i < N; i++)
		mu += i*h[i];

	mu *= scale;
	double mu1 = 0, q1 = 0;
	double max_sigma = 0, max_val = 0;

	for (i = 0; i < N; i++)
	{
		double p_i, q2, mu2, sigma;

		p_i = h[i] * scale;
		mu1 *= q1;
		q1 += p_i;
		q2 = 1. - q1;

		if (MIN(q1, q2) < FLT_EPSILON || max_my(q1, q2) > 1. - FLT_EPSILON)
			continue;

		mu1 = (mu1 + i*p_i) / q1;
		mu2 = (mu - q1*mu1) / q2;
		sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
		if (sigma > max_sigma)
		{
			max_sigma = sigma;
			max_val = i;
		}
	}

	return max_val;
}



bool createDetectorDescriptorMatcher(const string& detectorType, const string& descriptorType, const string& matcherType,
	Ptr<FeatureDetector>& featureDetector,
	Ptr<DescriptorExtractor>& descriptorExtractor,
	Ptr<DescriptorMatcher>& descriptorMatcher)
{
	cout << "< Creating feature detector, descriptor extractor and descriptor matcher ..." << endl;
	if (detectorType == "SIFT" || detectorType == "SURF")
		initModule_nonfree();
	featureDetector = FeatureDetector::create(detectorType);
	descriptorExtractor = DescriptorExtractor::create(descriptorType);
	descriptorMatcher = DescriptorMatcher::create(matcherType);
	cout << ">" << endl;
	bool isCreated = !(featureDetector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty());
	if (!isCreated)
		cout << "Can not create feature detector or descriptor extractor or descriptor matcher of given types." << endl << ">" << endl;
	return isCreated;
}

static std::vector<std::vector<cv::Point>> find1(Mat image)
{
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(image,
		contours, // a vector of contours   
		CV_RETR_LIST,//CV_RETR_EXTERNAL,//CV_RETR_TREE, // retrieve the external contours  
		CV_LINK_RUNS);//CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours  

	return contours;

}

int find(int set[], int x)
{
	int r = x;
	while (set[r] != r)
		r = set[r];
	return r;
}


void gray_projection(vector<int> &Sx, vector<int>&Sy, Mat I)
{
	int m = I.rows;
	int n = I.cols;
	for (int i = 0; i < m; i++)
	{

		for (int j = 0; j < n; j++)
			Sx[i] += I.at<uchar>(i, j);// / (abs(n / 2 - j) + 1);
		//Sx[i] = Sx[i]/n ;

	}
	//Sx[0] = 0; Sx[m - 1] = 0;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
			Sy[i] += I.at<uchar>(j, i);// / (abs(m / 2 - j) + 1);
		//Sy[i] = Sy[i]/m ;
	}
	//Sy[0] = 0; Sy[n - 1] = 0;

}




void quick_sort(double *s, vector<Mat> &H_list, int l, int r)
{
	if (l < r)
	{
		int i = l, j = r;
		double x = s[l];
		Mat temp = H_list[l].clone();
		while (i < j)
		{
			while (i < j && s[j] >= x) // ´ÓÓÒÏò×óÕÒµÚÒ»¸öÐ¡ÓÚxµÄÊý  
				j--;
			if (i < j)
			{
				s[i++] = s[j];
				H_list[j].copyTo(H_list[i - 1]);
			}

			while (i < j && s[i] < x) // ´Ó×óÏòÓÒÕÒµÚÒ»¸ö´óÓÚµÈÓÚxµÄÊý  
				i++;
			if (i < j)
			{
				s[j--] = s[i];
				H_list[i].copyTo(H_list[j + 1]);
			}
		}
		s[i] = x;
		temp.copyTo(H_list[i]);
		quick_sort(s, H_list, l, i - 1); // µÝ¹éµ÷ÓÃ   
		quick_sort(s, H_list, i + 1, r);
	}
}

bool check_coefficients(Mat &H)
{
	//½µ²ÉÑù2±¶
	//ÇãÐ±¡¢·ÅËõ²ÎÊý¼ì²é
	if ((((double*)H.data)[1] > 0.1) || (((double*)H.data)[1] < -0.1) || (((double*)H.data)[3] > 0.1) || (((double*)H.data)[3] < -0.1) || \
		(((double*)H.data)[0] > 1.1) || (((double*)H.data)[0] < 0.9) || (((double*)H.data)[4] > 1.1) || (((double*)H.data)[4] < 0.9))
		return false;
	else
		return true;
}

double Residual(Mat H, Mat X1, Mat X2)
{
	int num = X1.cols;
	//ÆÀ¼ÛÎó²î
	Mat X2_ = H*X1;
	Mat X2_row_3 = Mat::zeros(3, num, CV_64F);
	X2_.row(2).copyTo(X2_row_3.row(0));
	X2_.row(2).copyTo(X2_row_3.row(1));
	X2_.row(2).copyTo(X2_row_3.row(2));
	X2_ /= X2_row_3;
	Mat dx = X2_.row(0) - X2.row(0);
	Mat dy = X2_.row(1) - X2.row(1);
	Mat d_x_y = (dx.mul(dx) + dy.mul(dy));
	//·µ»ØÖµerr
	double err = sum(d_x_y).val[0];

	return err;
}

void Nelder_Mead(Mat &H0, Mat &pt_bg_inlier, Mat &cor_smooth_inlier, int max_iter, double eps, Mat &H, bool show_best)
{
	const int Max_time = max_iter;
	//Í¸ÊÓ±ä»»¾ØÕó£¬¼´µ¥Ó¦ÐÔ¾ØÕóÊ±£¬ÓÐ8¸ö±äÁ¿
	int var_num = 8;
	vector<Mat> vx(var_num + 1);
	H0.copyTo(vx[0]);
	double vf[9] = { 0, 0, 0, 0, 0, 0, 0 };
	vf[0] = Residual(H0, pt_bg_inlier, cor_smooth_inlier);
	//Ö»½«µ¥Ó¦¾ØÕóµÄÇ°Á½ÐÐ´úÈë¼ÆËã
	//cout<<H0<<endl;
	for (int i = 0; i<3; i++)
	{
		for (int j = 0; j<3; j++)
		{
			if (!(i == 2 && j == 2))
			{
				H0.copyTo(vx[i * 3 + j + 1]);
				if ((fabs(((double*)H0.data)[i * 3 + j])) < 0.00005)	//Èç¹ûÌ«Ð¡£¬ÔòÈÏÎª¼ÓÉÏÒ»¸öºÜÐ¡µÄÈÅ¶¯
					((double*)vx[i * 3 + j + 1].data)[i * 3 + j] += 0.005;
				else
					((double*)vx[i * 3 + j + 1].data)[i * 3 + j] /= 1.05;		//·ñÔò£¬³ËÒÔÒ»¸öÏµÊý
				//²ÎÊýÏÞ¶¨
				//constrain_coefficients(vx[i*3+j+1]);
				//cout<<vx[i*3+j+1]<<endl;
				vf[i * 3 + j + 1] = Residual(vx[i * 3 + j + 1], pt_bg_inlier, cor_smooth_inlier);	//¼ÆËãÆä¶ÔÓ¦Îó²î
			}
		}
	}
	//ÅÅÐò
	quick_sort(vf, vx, 0, var_num);

	double max_of_this = 0;
	double max_err = 0;
	while (max_iter>0)
	{
		for (int i = 0; i<var_num + 1; i++)
		{
			for (int j = i + 1; j<var_num + 1; j++)
			{
				Mat abs_err = abs(vx[i] - vx[j]);
				for (int k = 0; k<3; k++)
				{
					if (((double*)abs_err.data)[k] > max_of_this)
						max_of_this = ((double*)abs_err.data)[k];
					if (((double*)abs_err.data)[k + 3] > max_of_this)
						max_of_this = ((double*)abs_err.data)[k + 3];
				}
				if (max_of_this > max_err)
					max_err = max_of_this;
			}
		}
		//max_err = fabs(vf[0] - vf[var_num]);
		if (show_best && max_iter % 100 == 0)
		{
			if (max_iter % 100 == 0)
				cout << max_err << "\t";
		}
		//Èç¹û¸÷¸ö²ÎÊýµÄ×î´óÎó²î×ã¹»Ð¡£¬ÔòÌø³öÑ­»·
		//ÓÐÊ±ºò£¬¹ì¼£Êý±È½ÏÉÙ£¬2*pt_bg_inlier.cols¾Í±È½ÏÐ¡£¬ÊÕÁ²Ìõ¼þ¾Í»áÌ«¿Á¿Ì
		if (max_err < eps && (vf[0] <= 50))
		{
			if (show_best)
			{
				cout << "µü´ú´ÎÊý:" << Max_time - max_iter << endl;
				cout << "×î´ó×îÐ¡Ïà²îÎª:" << max_err << endl;
				cout << "ÒÑÕÒµ½×îÓÅ½á¹û£¬×îÐ¡Ïà²îÎª:" << vf[0] << endl;
			}
			break;
		}
		//Ëã·¨ºËÐÄÄ£¿é
		Mat best = vx[0];
		double fbest = vf[0];
		Mat soso = vx[var_num - 1];
		double fsoso = vf[var_num - 1];
		Mat worst = vx[var_num];
		double fworst = vf[var_num];
		Mat center = Mat::zeros(3, 3, CV_64F);
		for (int i = 0; i<var_num; i++)
			center += vx[i];
		center /= var_num;
		Mat r = 2 * center - worst;
		//²ÎÊýÏÞ¶¨
		//constrain_coefficients(r);
		double fr = Residual(r, pt_bg_inlier, cor_smooth_inlier);
		if (fr < fbest)
		{
			//±È×îºÃµÄ½á¹û»¹ºÃ£¬ËµÃ÷·½ÏòÕýÈ·£¬¿¼²ìÀ©Õ¹µã£¬ÒÔÆÚÍû¸ü¶àµÄÏÂ½µ
			Mat e = 2 * r - center;
			//²ÎÊýÏÞ¶¨
			//constrain_coefficients(e);
			double fe = Residual(e, pt_bg_inlier, cor_smooth_inlier);
			//ÔÚÀ©Õ¹µãºÍ·´ÉäµãÖÐÑ¡Ôñ½ÏÓÅÕßÈ¥Ìæ»»×î²îµã
			if (fe < fr)
			{
				vx[var_num] = e;//e.clone();
				vf[var_num] = fe;
			}
			else
			{
				vx[var_num] = r;//r.clone();
				vf[var_num] = fr;
			}
		}
		else
		{
			if (fr < fsoso)
			{
				//±È´Î²î½á¹ûºÃ£¬ÄÜ¸Ä½ø
				vx[var_num] = r;//r.clone();
				vf[var_num] = fr;
			}
			else//±È´Î²î½á¹û»¹²î£¬Ó¦¿¼ÂÇÑ¹Ëõµã
			{
				//µ±Ñ¹ËõµãÎÞ·¨µÃµ½¸üÓÅÖµµÄÊ±ºò£¬¿¼ÂÇÊÕËõ
				bool shrink = false;
				if (fr < fworst)
				{
					//ÓÉÓÚrµã¸üÓÅ£¬ËùÒÔÏòrµãµÄ·½ÏòÕÒÑ¹Ëõµã
					Mat c = (r + center) / 2;
					//²ÎÊýÏÞ¶¨
					//constrain_coefficients(c);
					double fc = Residual(c, pt_bg_inlier, cor_smooth_inlier);
					if (fc < fr)
					{
						//È·¶¨´ÓrÑ¹ËõÏòc¿ÉÒÔ¸Ä½ø
						vx[var_num] = c;//c.clone();
						vf[var_num] = fc;
					}
					else
						//·ñÔòµÄ»°£¬×¼±¸½øÐÐÊÕËõ
						shrink = true;
				}
				else
				{
					//ÓÉÓÚwµã¸üÓÅ£¬ËùÒÔÏòwµãµÄ·½ÏòÕÒÑ¹Ëõµã
					Mat c = (worst + center) / 2;
					//²ÎÊýÏÞ¶¨
					//constrain_coefficients(c);
					double fc = Residual(c, pt_bg_inlier, cor_smooth_inlier);
					if (fc < fr)
					{
						//È·¶¨´ÓrÑ¹ËõÏòc¿ÉÒÔ¸Ä½ø
						vx[var_num] = c;//c.clone();
						vf[var_num] = fc;
					}
					else
						//·ñÔòµÄ»°£¬×¼±¸½øÐÐÊÕËõ
						shrink = true;
				}
				if (shrink)
				{
					for (int i = 1; i<var_num + 1; i++)
					{
						Mat temp = (vx[i] + best) / 2;
						//²ÎÊýÏÞ¶¨
						//constrain_coefficients(temp);
						vx[i] = temp;//temp.clone();
						vf[i] = Residual(vx[i], pt_bg_inlier, cor_smooth_inlier);
					}
				}
			}
		}
		//ÅÅÐò
		quick_sort(vf, vx, 0, var_num);
		//if(max_iter>900)
		//	cout<<"×îÐ¡Îó²îÊÇ"<<vf[0]<<endl;
		max_iter--;
	}
	H = vx[0].clone();
	//cout<<"×îÓÅ½á¹û"<<H<<endl;
	//cout<<"×îÐ¡Îó²îÊÇ"<<vf[0]<<endl;
	//cout<<Residual(H0, pt_bg_inlier, cor_smooth_inlier);
}

Mat cal_cor(Mat H, Mat point1)
{
	Mat point2(3, 1, CV_64FC1);
	point2.row(0).col(0) = int((H.at<double>(0, 0)*(point1.at<double>(0, 0)) + H.at<double>(0, 1) * point1.at<double>(0, 1) + H.at<double>(0, 2)) / (H.at<double>(2, 0)*point1.at<double>(0, 0) + H.at<double>(2, 1) *point1.at<double>(1, 0) + H.at<double>(2, 2)));
	point2.row(1).col(0) = int((H.at<double>(1, 0)*(point1.at<double>(0, 0)) + H.at<double>(1, 1) * point1.at<double>(0, 1) + H.at<double>(1, 2)) / (H.at<double>(2, 0)*point1.at<double>(0, 0) + H.at<double>(2, 1) *point1.at<double>(1, 0) + H.at<double>(2, 2)));
	point2.row(2).col(0) = 1;
	return point2;
}



Mat Homography_Nelder_Mead_with_outliers(vector<Point2d> &pt_bg_cur, vector<Point2d> &Trj_cor_smooth, int max_iter/*, Mat& outliers, int height*/)
{
	//int64 st, et;
	//st = cvGetTickCount();
	int RANSAC_times = 500;
	double thresh_inlier = 1;// 25 / ((720.0 / height)*(720.0 / height));//80/(scale*scale);
	int num = pt_bg_cur.size();
	//¹¹Ôì¹éÒ»»¯×ø±êÏòÁ¿µÄ¾ØÕó
	Mat pt_bg = Mat::ones(3, num, CV_64F), cor_smooth = Mat::ones(3, num, CV_64F);
	//ÓÃÓÚ²úÉúËæ»úÐòÁÐ
	vector<int>index_shuffle(num);
	for (int i = 0; i<num; i++)
	{
		index_shuffle[i] = i;
		((double*)pt_bg.data)[i] = pt_bg_cur[i].x;
		((double*)pt_bg.data)[i + num] = pt_bg_cur[i].y;
		((double*)cor_smooth.data)[i] = Trj_cor_smooth[i].x;
		((double*)cor_smooth.data)[i + num] = Trj_cor_smooth[i].y;
	}

	//RANSACËã·¨£¬×î¶à100´ÎÑ­»·
	srand((unsigned)time(0));
	Mat OK = Mat::zeros(RANSAC_times, num, CV_8U);			//ºÃµÄ½á¹û£¬1±íÊ¾¸ÃÊý¾ÝÓëÄ£ÐÍÆ¥ÅäµÃºÃ£¬0Îª²»ºÃ
	vector<int> Score(RANSAC_times);								//ÆÀ¼ÛÎó²îµÃ·Ö£¬µÃ·ÖÔ½¸ß±íÊ¾Ä£ÐÍÔ½ºÃ
	vector<Mat> H(RANSAC_times);									//Ã¿´ÎµÄµ¥Ó¦¾ØÕó
	vector<double> Total_err(RANSAC_times);						//×ÜÌåÎó²î
	Mat thresh = thresh_inlier*Mat::ones(1, num, CV_64F);
	Mat every_outliers = Mat::zeros(RANSAC_times, num, CV_8U);
	int best_index = -1;		//×îºÃÄ£ÐÍµÄË÷ÒýÖµ
	int best = -1;				//ScoreµÄ×î´óÖµ
	//ËÑ²»µ½ÔÚºÏÊÊ·¶Î§ÄÚµÄ×îÓÅÖµ£¬¾ÍÔÙÑ­»·Ò»´Î
	while (best == -1)
	{
		for (int t = 0; t<RANSAC_times; t++)
		{
			//Ëæ»ú³éÈ¡ËÄ¸öµã£¬¹¹Ôì×ó±ßA¾ØÕó
			vector<int> rand_set;
			//ÏÈÓÃshuffleËã·¨Éú³ÉËæ»úÐòÁÐ
			random_shuffle(index_shuffle.begin(), index_shuffle.end(), myrandom);
			rand_set.push_back(index_shuffle[0]);
			rand_set.push_back(index_shuffle[1]);
			rand_set.push_back(index_shuffle[2]);
			rand_set.push_back(index_shuffle[3]);

			//AÒ»¶¨ÒªÉè¶¨Îª¾Ö²¿±äÁ¿£¡£¡£¡ÒòÎªÏÂÃæÊ¹ÓÃ+=£¬¶ø²»ÊÇ¸³Öµ=£¡£¡£¡£¡
			Mat A = Mat::zeros(12, 9, CV_64F);
			int j = 0;
			int k = rand_set[0];	//0 <= k < num
			Mat hat = (Mat_<double>(3, 3) << 0, -1, ((double*)cor_smooth.data)[k + num], 1, 0, -1 * ((double*)cor_smooth.data)[k], -1 * ((double*)cor_smooth.data)[k + num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			double x = ((double*)pt_bg.data)[k];
			double y = ((double*)pt_bg.data)[k + num];
			A.rowRange(j * 3, j * 3 + 3).colRange(0, 3) += hat*x;
			A.rowRange(j * 3, j * 3 + 3).colRange(3, 6) += hat*y;
			A.rowRange(j * 3, j * 3 + 3).colRange(6, 9) += hat;

			++j;
			k = rand_set[1];	//0 <= k < num
			hat = (Mat_<double>(3, 3) << 0, -1, ((double*)cor_smooth.data)[k + num], 1, 0, -1 * ((double*)cor_smooth.data)[k], -1 * ((double*)cor_smooth.data)[k + num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((double*)pt_bg.data)[k];
			y = ((double*)pt_bg.data)[k + num];
			A.rowRange(j * 3, j * 3 + 3).colRange(0, 3) += hat*x;
			A.rowRange(j * 3, j * 3 + 3).colRange(3, 6) += hat*y;
			A.rowRange(j * 3, j * 3 + 3).colRange(6, 9) += hat;

			++j;
			k = rand_set[2];	//0 <= k < num
			hat = (Mat_<double>(3, 3) << 0, -1, ((double*)cor_smooth.data)[k + num], 1, 0, -1 * ((double*)cor_smooth.data)[k], -1 * ((double*)cor_smooth.data)[k + num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((double*)pt_bg.data)[k];
			y = ((double*)pt_bg.data)[k + num];
			A.rowRange(j * 3, j * 3 + 3).colRange(0, 3) += hat*x;
			A.rowRange(j * 3, j * 3 + 3).colRange(3, 6) += hat*y;
			A.rowRange(j * 3, j * 3 + 3).colRange(6, 9) += hat;

			//Mat temp = A.rowRange(j*3, j*3+3).colRange(0,3).clone();

			++j;
			k = rand_set[3];	//0 <= k < num
			hat = (Mat_<double>(3, 3) << 0, -1, ((double*)cor_smooth.data)[k + num], 1, 0, -1 * ((double*)cor_smooth.data)[k], -1 * ((double*)cor_smooth.data)[k + num], ((double*)cor_smooth.data)[k], 0);
			//cout<<hat<<endl;
			x = ((double*)pt_bg.data)[k];
			y = ((double*)pt_bg.data)[k + num];
			A.rowRange(j * 3, j * 3 + 3).colRange(0, 3) += hat*x;
			A.rowRange(j * 3, j * 3 + 3).colRange(3, 6) += hat*y;
			A.rowRange(j * 3, j * 3 + 3).colRange(6, 9) += hat;

			//cout<<A<<endl;
			//ofstream A_file("A_file.txt");
			//A_file<<A<<endl;
			//SVD·Ö½âÉú³ÉVT£¬µÚ9ÐÐÎª×îÐ¡ÌØÕ÷Öµ¶ÔÓ¦µÄÌØÕ÷ÏòÁ¿
			SVD thissvd(A, SVD::FULL_UV);
			Mat VT = thissvd.vt;
			//cout<<VT<<endl;
			//Éú³É±¾´ÎRANSACÑ­»·¶ÔÓ¦µÄ¹éÒ»»¯µÄµ¥Ó¦¾ØÕó
			H[t] = (Mat_<double>(3, 3) << ((double*)VT.data)[72], ((double*)VT.data)[75], ((double*)VT.data)[78], ((double*)VT.data)[73], ((double*)VT.data)[76], ((double*)VT.data)[79], ((double*)VT.data)[74], ((double*)VT.data)[77], ((double*)VT.data)[80]);// / ((double*)VT.data)[80];
			//cout<<H[t]<<endl;
			H[t] /= ((double*)H[t].data)[8];
			//cout<<H[t]<<endl;

			//ÆÀ¼ÛÎó²î
			Mat X2_ = H[t] * pt_bg;
			//cout<<X2_<<endl;
			Mat X2_row_3 = Mat::zeros(3, num, CV_64F);
			X2_.row(2).copyTo(X2_row_3.row(0));
			X2_.row(2).copyTo(X2_row_3.row(1));
			X2_.row(2).copyTo(X2_row_3.row(2));
			X2_ /= X2_row_3;
			//cout<<X2_<<endl;
			Mat dx = X2_.row(0) - cor_smooth.row(0);
			Mat dy = X2_.row(1) - cor_smooth.row(1);
			Mat d_x_y = (dx.mul(dx) + dy.mul(dy));
			//cout<<d_x_y<<endl;
			//½á¹û¼ÇÂ¼ÔÚTotal_err¡¢OKºÍScore¾ØÕóÖÐ
			Total_err[t] = sum(d_x_y).val[0];
			OK.row(t) = (d_x_y < thresh) / 255.f;
			//cout<<OK.row(t)<<endl;
			Scalar sum_o = sum(OK.row(t));
			Score[t] = sum(OK.row(t)).val[0];
			//¼ÇÂ¼×îºÃ½á¹ûµÄË÷ÒýÖµ
			if (Score[t] > best)
			{
				//cout<<H[t]<<endl;
				if (check_coefficients(H[t]))
				{
					best = Score[t];
					best_index = t;
				}
			}
			else if (Score[t] == best)	//Ä£ÐÍÆ¥ÅäÊýÁ¿Ò»ÖÂÊ±£¬È¡Îó²î×îÐ¡µÄ
			{
				if (Total_err[t] < Total_err[best_index])
				{
					if (check_coefficients(H[t]))
					{
						best = Score[t];
						best_index = t;
					}
				}
			}
		}
	}
	//et = cvGetTickCount();
	//printf("RANSACÑ­»·£¬100´ÎÊ±¼äÎª: %f\n", (et-st)/(double)cvGetTickFrequency()/1000.);
	//cout<<"Æ¥ÅäÉÏÁË"<<best<<"¸ö"<<endl;
	//outliers = OK.row(best_index).clone();
	//cout<<outliers<<endl;
	//ÌáÈ¡³öÄÚµã
	Mat pt_bg_inlier = Mat::zeros(3, best, CV_64F), cor_smooth_inlier = Mat::zeros(3, best, CV_64F);
	int inlier_ind = 0;
	//cout<<"OK.row(best_index)"<<endl;
	//cout<<OK.row(best_index)<<endl;
	for (int i = 0; i<num; i++)
	{
		if (((unsigned char*)OK.data)[best_index*num + i] > 0)
		{
			pt_bg.col(i).copyTo(pt_bg_inlier.col(inlier_ind));
			cor_smooth.col(i).copyTo(cor_smooth_inlier.col(inlier_ind));
			inlier_ind++;
		}
	}
	//Nelder-MeadËã·¨ËÑË÷×îÓÅÖµ
	//Ç¿ÖÆµ¥Ó¦¾ØÕó±äÎª·ÂÉä¾ØÕó£¬¼´µÚ3ÐÐÇ°Á½¸öÔªËØÎª0
	Mat H0 = H[best_index];
	//Mat H_best = Mat::zeros(3, 3, CV_64F);
	//H0.row(2).col(0) = 0.f;
	//H0.row(2).col(1) = 0.f;
	Mat H_NM = Mat::zeros(3, 3, CV_64F);
	double eps = 0.001;
	//st = cvGetTickCount();
	bool show_ransac = false;
	Nelder_Mead(H0, pt_bg_inlier, cor_smooth_inlier, max_iter, eps, H_NM, show_ransac);
	//et = cvGetTickCount();
	//printf("NMËÑË÷Ê±¼ä: %f\n", (et-st)/(double)cvGetTickFrequency()/1000.);
	return H_NM;
}
Mat hist_extend(Mat src, int &flag, int range)
{
	int min = INT_MAX;
	int max = 0;
	Mat dst = src.clone();
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (min>src.at<uchar>(i, j))
				min = src.at<uchar>(i, j);
			if (max<src.at<uchar>(i, j))
				max = src.at<uchar>(i, j);
		}
	if (max - min > 127 || max - min<10)
		flag = 1;
	else
		flag = 0;
	//cout << max - min << endl;
	if (max - min <= 200 && max - min >= 10)
	{
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
			{
				//dst.row(i).col(j) = MIN(int(double(src.at<uchar>(i, j) - min)*2),255);// / double(max - min) * 255);
				dst.row(i).col(j) = int((double(src.at<uchar>(i, j)) - min) / double(max - min)*range);
			}
	}
	return dst;
}

double sum_diff(Mat src, Mat dst)
{
	//src = hist_extend(src);
	//dst = hist_extend(dst);
	double mean1 = 0;
	double mean2 = 0;

	for (int i = 0; i < src.size().height; i++)
		for (int j = 0; j < src.size().width; j++)
		{
			mean1 += double(src.at<uchar>(i, j));
			mean2 += double(dst.at<uchar>(i, j));
		}
	mean1 = mean1 / (src.size().width*src.size().height);
	mean2 = mean2 / (src.size().width*src.size().height);
	double sum = 0;
	for (int i = 0; i < src.size().height; i++)
		for (int j = 0; j< src.size().width; j++)
		{
			sum += abs(double(src.at<uchar>(i, j)) - mean1 - double(dst.at<uchar>(i, j)) + mean2);
		}
	sum = sum / (src.size().height*src.size().width);
	return sum;
}
void copy_im(Mat &src_result, int x, int y, int h, int w, int gray_diff_xx2, int gray_diff_yy2, Mat src_select, Mat dst_select)
{

	Mat  result = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
	Mat src_temp = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
	Mat dst_temp = dst_select.rowRange(gray_diff_yy2, gray_diff_yy2 + h).colRange(gray_diff_xx2, gray_diff_xx2 + w).clone();
	double mean1 = 0;
	double mean2 = 0;

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
		{
			mean1 += double(src_temp.at<uchar>(i, j));
			mean2 += double(dst_temp.at<uchar>(i, j));
		}
	mean1 = mean1 / (h*w);
	mean2 = mean2 / (h*w);
	double max = 0;

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
		{
			if (max <abs(double(src_temp.at<uchar>(i, j)) - mean1 - double(dst_temp.at<uchar>(i, j)) + mean2))
				max = abs(double(src_temp.at<uchar>(i, j)) - mean1 - double(dst_temp.at<uchar>(i, j)) + mean2);
		}
	int min_src = INT_MAX, max_src = 0, min_dst = INT_MAX, max_dst = 0;
	for (int i = 0; i < src_temp.rows; i++)
		for (int j = 0; j < src_temp.cols; j++)
		{
			if (min_src>src_temp.at<uchar>(i, j))
				min_src = src_temp.at<uchar>(i, j);
			if (max_src<src_temp.at<uchar>(i, j))
				max_src = src_temp.at<uchar>(i, j);
		}
	for (int i = 0; i < src_temp.rows; i++)
		for (int j = 0; j < src_temp.cols; j++)
		{
			if (min_dst>dst_temp.at<uchar>(i, j))
				min_dst = dst_temp.at<uchar>(i, j);
			if (max_dst<dst_temp.at<uchar>(i, j))
				max_dst = dst_temp.at<uchar>(i, j);
		}
	int range = MAX(max_src - min_src, max_dst - min_dst);

	int flag1 = 0, flag2 = 0;

	while (max > 30 && flag1*flag2 == 0)
	{
		range = MIN(range * 2, 255);
		src_temp = hist_extend(src_temp, flag1, range);
		dst_temp = hist_extend(dst_temp, flag2, range);
		mean1 = 0;
		mean2 = 0;

		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
			{
				mean1 += double(src_temp.at<uchar>(i, j));
				mean2 += double(dst_temp.at<uchar>(i, j));
			}
		mean1 = mean1 / (h*w);
		mean2 = mean2 / (h*w);
		double max = 0;

		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
			{
				if (max <abs(double(src_temp.at<uchar>(i, j)) - mean1 - double(dst_temp.at<uchar>(i, j)) + mean2))
					max = abs(double(src_temp.at<uchar>(i, j)) - mean1 - double(dst_temp.at<uchar>(i, j)) + mean2);
			}
	}
	mean1 = 0;
	mean2 = 0;

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
		{
			mean1 += double(src_temp.at<uchar>(i, j));
			mean2 += double(dst_temp.at<uchar>(i, j));
		}
	mean1 = mean1 / (h*w);
	mean2 = mean2 / (h*w);
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
		{
			result.row(i).col(j) = abs(double(src_temp.at<uchar>(i, j)) - mean1 - double(dst_temp.at<uchar>(i, j)) + mean2);
		}
	result.copyTo(src_result.rowRange(y, y + h).colRange(x, x + w));
}

void cocy_rgb(Mat &src_result, int x, int y, int h, int w, int gray_diff_xx2, int gray_diff_yy2, Mat src_select, Mat dst_select)
{
	//Mat  result = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
	Mat result(h, w, CV_8UC1);
	Mat src_temp = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
	Mat dst_temp = dst_select.rowRange(gray_diff_yy2, gray_diff_yy2 + h).colRange(gray_diff_xx2, gray_diff_xx2 + w).clone();
	Mat mv_src[3], mv_dst[3];
	split(src_temp, mv_src);
	split(dst_temp, mv_dst);
	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
		{
			int flag = -1;
			int max = -1;
			for (int p = 0; p < 3; p++)
			{
				int temp = abs(mv_src[p].at<uchar>(i, j) - mv_dst[p].at<uchar>(i, j));
				if (temp > max)
					max = temp;
			}
			result.row(i).col(j) = max;
		}
	result.copyTo(src_result.rowRange(y, y + h).colRange(x, x + w));
}

double compute_mean(Mat src)
{
	double sum = 0;
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			sum += double(src.at<uchar>(i, j));
		}
	return sum / (src.rows*src.cols);
}


//int main()
//{
extern "C" __declspec(dllexport) int __stdcall textdetect(const char* srcc, const char* dstt, const char* path, int x, int y, int w, int h, const char* matrix, int warp)
{
	//string srcc = "F:\\developfiles\\DifferentialDetection\\DifferentialDetection\\bin\\x64\\Release\\templates\\1\\289630-Ô­¸å1-ps1.jpg";
	//string dstt = "F:\\developfiles\\DifferentialDetection\\DifferentialDetection\\bin\\x64\\Release\\templates\\1\\detections\\1\\20170729-075-ps2.jpg";

	//string path = "F:\\developfiles\\DifferentialDetection\\DifferentialDetection\\bin\\x64\\Release\\templates\\1\\detections\\1\\";
	//int x = 0;
	//int y = 0;
	//int w = 1961;
	//int h = 1224;

	//string matrix = "1.3275 0.0118491 -29.4903,-0.00762689 1.33191 48.9623,5.08049e-007 1.2535e-006 1;";
	//int warp_i = 0;


	//SYSTEMTIME st = { 0 };
	//GetLocalTime(&st);
	//if (!(st.wYear == 2017 || (st.wYear == 2018 && st.wMonth<7)))
	//	return -1;

	cout << "text detect" << endl;
	DWORD dwRetCode;
	DWORD dwCount;
	WORD Index;
	DWORD dwHID;
	dwRetCode = VikeyFind(&dwCount);
	if (dwRetCode)
	{
		printf("\nERROR: Please Insert the Key! \n");
		return -1;
	}

	Index = 0;

	dwRetCode = VikeyGetHID(Index, &dwHID);
	if (dwRetCode)
	{
		printf("\nERROR: Failed to Get the KeyID! \n");
		return -1;
	}


	int hid = (int)dwHID;
	int hid_10 = 0;
	for (int i = 0; i < 9; i++)
	{
		hid_10 += hid % 10;
		hid /= 10;
	}








	vector<Rect>result_rect_src;
	vector<Rect>result_rect_dst;
	Mat src_whole;
	Mat dst_whole;
	Mat src_hsv;
	Mat dst_hsv;
	vector<Mat> channels_src;
	vector<Mat> channels_dst;
	/*src_whole = imread("pan1.jpg");
	dst_whole = imread("pan2.jpg");*/
	char UserPassWord[8];
	string src_img = srcc;
	string dst_img = dstt;
	string p = path;
	UserPassWord[0] = 'u';

	cout << "src: " << srcc << endl << "dst: " << dstt << endl;
	cout << "path: " << p << endl;
	cout << "mat: " << matrix << endl;
	cout << "w: " << w << " h: " << h << endl;
	UserPassWord[1] = 's';
	src_whole = imread(src_img);
	dst_whole = imread(dst_img);
	//string p = "d://";
	//int x_label = 0;
	//int y_label = 0;
	//int w_label = 3996;//2677;
	//int h_label = 3346;//1000
	UserPassWord[2] = 't';
	int x_label = x;
	int y_label = y;
	int w_label = w;
	int h_label = h;
	UserPassWord[3] = 'c';
	vector<Mat>warp_H;
	string mats = matrix;
	istringstream iss(mats);
	string sub;
	UserPassWord[4] = '0';
	
	while (getline(iss, sub, ';')) {
		istringstream isssub(sub);
		vector<string> locs;
		string sssub;
		Mat temp_H(3, 3, CV_64FC1);
		for (int i = 0; i < 3; i++){
			getline(isssub, sssub, ',');
			istringstream tinyiss(sssub);
			string tinysub;
			for (int j = 0; j < 3; j++){
				getline(tinyiss, tinysub, ' ');
				stringstream ss;
				double s;
				ss << tinysub;
				ss >> s;
				temp_H.row(i).col(j) = s;
			}
		}
		warp_H.push_back(temp_H);
	}
	
	/*Mat temp_H(3, 3, CV_64FC1);

	temp_H.row(0).col(0) = 0.999;
	temp_H.row(1).col(1) = 1;
	temp_H.row(2).col(2) = 1;
	temp_H.row(0).col(1) = 0;
	temp_H.row(0).col(2) = -0.222;
	temp_H.row(1).col(0) = 0;
	temp_H.row(1).col(2) = 0.517;
	temp_H.row(2).col(0) = 0;
	temp_H.row(2).col(1) = 0;*/


	/*warp_H.push_back(temp_H);*/
	m_RecInfoCall(30);
	double progress = 0;
	UserPassWord[5] = '0';
	double delta = 0.65 / warp_H.size();
	cout << "warp_H.size:" << warp_H.size() << endl;




	UserPassWord[6] = '3';
	UserPassWord[7] = '0';
	/////////////////////////////////////////////
	if (warp_H.size() == 1){

		cout << "warp_H[0]" << endl << warp_H[0] << endl;
		Mat imageturn = Mat::zeros(src_whole.size().width, src_whole.size().height, src_whole.type());
		cout << "1" << endl;
		warpPerspective(dst_whole, imageturn, warp_H[0].inv(), cv::Size(imageturn.rows, imageturn.cols));
		cout << "2" << endl;
		Mat src = src_whole.rowRange(y_label, y_label + h_label).colRange(x_label, x_label + w_label).clone();
		cout << "3" << endl;
		Mat dst = imageturn.rowRange(y_label, y_label + h_label).colRange(x_label, x_label + w_label).clone();
		cout << "4" << endl;

		imwrite(p + "dst0.jpg", dst);

		std::cout << warp_H[0] << endl;

		progress = 0.05;



		Mat src_select, dst_select;
		Mat src1, dst1, src_result;

		cv::cvtColor(src, src_result, CV_BGR2GRAY);

		progress = 0.1;

		m_RecInfoCall(35);

		cvtColor(src, src_select, CV_BGR2GRAY);
		cvtColor(dst, dst_select, CV_BGR2GRAY);
		GaussianBlur(src_select, src_select, Size(3, 3), 0, 0);
		GaussianBlur(dst_select, dst_select, Size(3, 3), 0, 0);
		GaussianBlur(src, src, Size(3, 3), 0, 0);
		GaussianBlur(dst, dst, Size(3, 3), 0, 0);
		double src_mean = compute_mean(src_select);
		double dst_mean = compute_mean(dst_select);
		cout << "src_mean=" << src_mean << endl;;
		cout << "dst_mean=" << dst_mean << endl;
		/////////////////////////////»ñÈ¡¶þÖµ»¯ºóµÄÐ¡¾ØÐÎ///////////////////////////
		vector<KeyPoint>kp1, kp2;

		int maxcorners = 5000;
		double qualitylevel = 0.01;
		double mindistance = 5;
		Mat mask1, mask2;
		GoodFeaturesToTrackDetector detector(maxcorners, qualitylevel, mindistance);
		progress = 0.2;

		m_RecInfoCall(40);
		detector.detect(src_select, kp1, mask1);
		detector.detect(dst_select, kp2, mask2);

		FREAK descriptor_extractor;

		cout << "kp1_size=" << kp1.size() << endl;
		cout << "kp2_size==" << kp2.size() << endl;
		//Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create("SIFT");//´´½¨ÌØÕ÷ÏòÁ¿Éú³ÉÆ÷   
		Mat descriptor1, descriptor2;
		descriptor_extractor.compute(src_select, kp1, descriptor1);
		descriptor_extractor.compute(dst_select, kp2, descriptor2);
		vector<DMatch> matches;
		BruteForceMatcher<HammingLUT>matcher;
		matcher.match(descriptor1, descriptor2, matches);


		cout << "match_size=" << matches.size() << endl;
		Mat img_matches;
		drawMatches(src_select, kp1, dst_select, kp2, matches, img_matches);
		Mat img_matches_resize;
		resize(img_matches, img_matches_resize, Size(1600, 900), 0, 0, CV_INTER_LINEAR);
		//imshow("matches", img_matches_resize);
		//cvWaitKey(0);
		progress = 0.3;

		m_RecInfoCall(45);
		vector<Point2d>point_src, point_dst;
		for (int i = 0; i < matches.size(); i++)
		{
			Point2d pt_temp_train, pt_temp_query;
			pt_temp_query.x = kp1[matches[i].queryIdx].pt.x;
			pt_temp_query.y = kp1[matches[i].queryIdx].pt.y;
			pt_temp_train.x = kp2[matches[i].trainIdx].pt.x;
			pt_temp_train.y = kp2[matches[i].trainIdx].pt.y;
			point_src.push_back(pt_temp_query);
			point_dst.push_back(pt_temp_train);
		}
		dwRetCode = VikeyUserLogin(Index, UserPassWord);
		if (dwRetCode)
		{
			printf("\nERROR: No Permission to Use the Software! \n");
			return -1;
		}
		else
		{
			cout << "vikey success!";
		}
		progress = 0.3;
		src_result.setTo(0);
		//2ÖÐµÄµã=t*1ÖÐµÄµã;Ò²¾ÍÊÇ½«1±ä»¯µ½2ÖÐ
		//if (point_src.size()>20)
		//{
		//Mat T = Homography_Nelder_Mead_with_outliers(point_src, point_dst, 5000);
		Mat T = findHomography(point_src, point_dst, CV_RANSAC);
		cout << T << endl;
		progress = 0.4;
		Mat point1(3, 1, CV_64FC1);


		int parameter_kernel = 0;
		WORD wAddr = 0;
		WORD wLen = 128;
		BYTE buffer[1024];
		dwRetCode = VikeyReadData(Index, wAddr, wLen, buffer);
		if (dwRetCode)
		{
			printf("Failed to read data!\n");
			return -1;
		}
		else
		{

			parameter_kernel = buffer[65];
		}
		if (parameter_kernel != 0)
			parameter_kernel = hid_10^parameter_kernel;



		m_RecInfoCall(50);
		//parameter_kernel = 50;
		int block_h = src.size().height / parameter_kernel;// int(double(MIN(src.size().height, src.size().width)) / 1000 * 30);
		int block_w = src.size().width / parameter_kernel;
		for (int i = 0; i <= src.size().height - block_h; i = i + block_h)
			for (int j = 0; j <= src.size().width - block_w; j = j + block_w)
				//for (int i = 0; i < boundRect_src.size(); i++)
			{
				//cout << i << endl;
				double x = j;
				double y = i;
				double w = block_w;
				double h = block_h;

				Mat src_temp = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
				double centerx = x + w / 2;
				double centery = y + h / 2;
				int index_min = -1;
				double min = INT_MAX;

				point1.row(0).col(0) = x;
				point1.row(1).col(0) = y;
				point1.row(2).col(0) = 1.0;
				Mat point2;
				//if (index_min>=0)
				//Mat	point2 = TT[index_min] * point1;
				//else
				//point2 = T * point1;
				point2 = cal_cor(T, point1);
				double x2 = MIN(MAX(point2.at<double>(0, 0), 0), dst.size().width - w);
				double y2 = MIN(MAX(point2.at<double>(1, 0), 0), dst.size().height - h);
				int drift_x = 3; //int(MAX(3, MIN(0.3*w, 5)));
				int drift_y = 3;// int(MAX(3, MIN(0.3*h, 5)));
				//Ìí¼ÓÅÐ¶Ï//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//cout << "x1===" << x << " y1===" << y << endl;
				//cout << "x2===" << x2 << " y2===" << y2 << endl;

				if (x2 >= 0 && y2 >= 0 && x2 + w <= dst.size().width && y2 + h <= dst.size().height)
				{
					vector<double>gray_diff_min;
					vector<int>gray_diff_xx2;
					vector<int>gray_diff_yy2;
					for (int xx2 = x2 - drift_x; xx2 <= x2 + drift_x; xx2 = xx2 + 1)
						for (int yy2 = y2 - drift_y; yy2 <= y2 + drift_y; yy2 = yy2 + 1)
						{
							if (xx2 >= 0 && yy2 >= 0 && xx2 + w <= src.size().width && yy2 + h <= src.size().height)
							{
								Mat dst_temp = dst_select.rowRange(max_my(yy2, 0), max_my(yy2, 0) + h).colRange(max_my(xx2, 0), max_my(xx2, 0) + w).clone();

								gray_diff_min.push_back(sum_diff(src_temp, dst_temp));
								gray_diff_xx2.push_back(xx2);
								gray_diff_yy2.push_back(yy2);
							}
						}
					int index_min = calculate_min(gray_diff_min);
					if (index_min != -1)
					{
						gray_diff.push_back(gray_diff_min[index_min]);
						//cocy_rgb(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src, dst);
						copy_im(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src_select, dst_select);
					}
				}
			}
		//×îºóÒ»ÐÐ
		m_RecInfoCall(55);
		for (int j = 0; j <= src.size().width - block_w; j = j + block_w)
			//for (int i = 0; i < boundRect_src.size(); i++)
		{
			//cout << i << endl;
			double x = j;
			double y = src.size().height - block_h;
			double w = block_w;
			double h = block_h;

			Mat src_temp = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
			double centerx = x + w / 2;
			double centery = y + h / 2;
			int index_min = -1;
			double min = INT_MAX;

			point1.row(0).col(0) = x;
			point1.row(1).col(0) = y;
			point1.row(2).col(0) = 1.0;
			Mat point2;
			//if (index_min>=0)
			//Mat	point2 = TT[index_min] * point1;
			//else
			//point2 = T * point1;
			point2 = cal_cor(T, point1);
			double x2 = MIN(MAX(point2.at<double>(0, 0), 0), dst.size().width - w);
			double y2 = MIN(MAX(point2.at<double>(1, 0), 0), dst.size().height - h);
			int drift_x = 3; //int(MAX(3, MIN(0.3*w, 5)));
			int drift_y = 3;// int(MAX(3, MIN(0.3*h, 5)));
			//Ìí¼ÓÅÐ¶Ï//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//cout << "x1===" << x << " y1===" << y << endl;
			//cout << "x2===" << x2 << " y2===" << y2 << endl;

			if (x2 >= 0 && y2 >= 0 && x2 + w <= dst.size().width && y2 + h <= dst.size().height)
			{
				vector<double>gray_diff_min;
				vector<int>gray_diff_xx2;
				vector<int>gray_diff_yy2;
				for (int xx2 = x2 - drift_x; xx2 <= x2 + drift_x; xx2 = xx2 + 1)
					for (int yy2 = y2 - drift_y; yy2 <= y2 + drift_y; yy2 = yy2 + 1)
					{
						if (xx2 >= 0 && yy2 >= 0 && xx2 + w <= src.size().width && yy2 + h <= src.size().height)
						{
							Mat dst_temp = dst_select.rowRange(max_my(yy2, 0), max_my(yy2, 0) + h).colRange(max_my(xx2, 0), max_my(xx2, 0) + w).clone();

							gray_diff_min.push_back(sum_diff(src_temp, dst_temp));
							gray_diff_xx2.push_back(xx2);
							gray_diff_yy2.push_back(yy2);
						}
					}
				int index_min = calculate_min(gray_diff_min);
				if (index_min != -1)
				{
					gray_diff.push_back(gray_diff_min[index_min]);
					//cocy_rgb(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src, dst);
					copy_im(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src_select, dst_select);
				}
			}
		}
		m_RecInfoCall(60);
		//×îºóÒ»ÁÐ
		for (int i = 0; i <= src.size().height - block_h; i = i + block_h)
			//for (int i = 0; i < boundRect_src.size(); i++)
		{
			//cout << i << endl;
			double x = src.size().width - block_w;
			double y = i;
			double w = block_w;
			double h = block_h;

			Mat src_temp = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
			double centerx = x + w / 2;
			double centery = y + h / 2;
			int index_min = -1;
			double min = INT_MAX;

			point1.row(0).col(0) = x;
			point1.row(1).col(0) = y;
			point1.row(2).col(0) = 1.0;
			Mat point2;
			//if (index_min>=0)
			//Mat	point2 = TT[index_min] * point1;
			//else
			//point2 = T * point1;
			point2 = cal_cor(T, point1);
			double x2 = MIN(MAX(point2.at<double>(0, 0), 0), dst.size().width - w);
			double y2 = MIN(MAX(point2.at<double>(1, 0), 0), dst.size().height - h);
			int drift_x = 3; //int(MAX(3, MIN(0.3*w, 5)));
			int drift_y = 3;// int(MAX(3, MIN(0.3*h, 5)));
			//Ìí¼ÓÅÐ¶Ï//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//cout << "x1===" << x << " y1===" << y << endl;
			//cout << "x2===" << x2 << " y2===" << y2 << endl;

			if (x2 >= 0 && y2 >= 0 && x2 + w <= dst.size().width && y2 + h <= dst.size().height)
			{
				vector<double>gray_diff_min;
				vector<int>gray_diff_xx2;
				vector<int>gray_diff_yy2;
				for (int xx2 = x2 - drift_x; xx2 <= x2 + drift_x; xx2 = xx2 + 1)
					for (int yy2 = y2 - drift_y; yy2 <= y2 + drift_y; yy2 = yy2 + 1)
					{
						if (xx2 >= 0 && yy2 >= 0 && xx2 + w <= src.size().width && yy2 + h <= src.size().height)
						{
							Mat dst_temp = dst_select.rowRange(max_my(yy2, 0), max_my(yy2, 0) + h).colRange(max_my(xx2, 0), max_my(xx2, 0) + w).clone();

							gray_diff_min.push_back(sum_diff(src_temp, dst_temp));
							gray_diff_xx2.push_back(xx2);
							gray_diff_yy2.push_back(yy2);
						}
					}
				int index_min = calculate_min(gray_diff_min);
				if (index_min != -1)
				{
					gray_diff.push_back(gray_diff_min[index_min]);
					//cocy_rgb(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src, dst);
					copy_im(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src_select, dst_select);
				}
			}
		}

		double x = src.size().width - block_w;
		double y = src.size().height - block_h;
		double w = block_w;
		double h = block_h;

		Mat src_temp = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
		double centerx = x + w / 2;
		double centery = y + h / 2;
		int index_min = -1;
		double min = INT_MAX;

		point1.row(0).col(0) = x;
		point1.row(1).col(0) = y;
		point1.row(2).col(0) = 1.0;
		Mat point2;
		//if (index_min>=0)
		//Mat	point2 = TT[index_min] * point1;
		//else
		//point2 = T * point1;
		point2 = cal_cor(T, point1);
		double x2 = MIN(MAX(point2.at<double>(0, 0), 0), dst.size().width - w);
		double y2 = MIN(MAX(point2.at<double>(1, 0), 0), dst.size().height - h);
		int drift_x = 3; //int(MAX(3, MIN(0.3*w, 5)));
		int drift_y = 3;// int(MAX(3, MIN(0.3*h, 5)));
		//Ìí¼ÓÅÐ¶Ï//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//cout << "x1===" << x << " y1===" << y << endl;
		//cout << "x2===" << x2 << " y2===" << y2 << endl;

		if (x2 >= 0 && y2 >= 0 && x2 + w <= dst.size().width && y2 + h <= dst.size().height)
		{
			vector<double>gray_diff_min;
			vector<int>gray_diff_xx2;
			vector<int>gray_diff_yy2;
			for (int xx2 = x2 - drift_x; xx2 <= x2 + drift_x; xx2 = xx2 + 1)
				for (int yy2 = y2 - drift_y; yy2 <= y2 + drift_y; yy2 = yy2 + 1)
				{
					if (xx2 >= 0 && yy2 >= 0 && xx2 + w <= src.size().width && yy2 + h <= src.size().height)
					{
						Mat dst_temp = dst_select.rowRange(max_my(yy2, 0), max_my(yy2, 0) +

							h).colRange(max_my(xx2, 0), max_my(xx2, 0) + w).clone();

						gray_diff_min.push_back(sum_diff(src_temp, dst_temp));
						gray_diff_xx2.push_back(xx2);
						gray_diff_yy2.push_back(yy2);
					}
				}
			int index_min = calculate_min(gray_diff_min);
			if (index_min != -1)
			{
				gray_diff.push_back(gray_diff_min[index_min]);
				//cocy_rgb(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src, dst);
				copy_im(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src_select, dst_select);
			}
		}
		imwrite(p + "result_base0.jpg", src_result);
		//double thres_src = getThreshVal_Otsu_8u(src_result);
	}


	/////////////////////////////////////////////
	else if (warp_H.size() > 1){
		for (int warp_i = 0; warp_i < warp_H.size(); warp_i++)
		{
			Mat imageturn = Mat::zeros(src_whole.size().width, src_whole.size().height, src_whole.type());
			warpPerspective(dst_whole, imageturn, warp_H[warp_i].inv(), cv::Size(imageturn.rows, imageturn.cols));
			Mat src = src_whole.rowRange(y_label, y_label + h_label).colRange(x_label, x_label + w_label).clone();
			Mat dst = imageturn.rowRange(y_label, y_label + h_label).colRange(x_label, x_label + w_label).clone();

			stringstream ss;
			ss << warp_i;

			imwrite(p + "dst" + ss.str() + ".jpg", dst);

			cout << warp_H[warp_i] << endl;

			progress = 0.05;



			Mat src_select, dst_select;
			Mat src1, dst1, src_result;

			cvtColor(src, src_result, CV_BGR2GRAY);

			progress = 0.1;

			m_RecInfoCall(30 + 33 / warp_H.size() * warp_i + 33 / warp_H.size() * 1 / 6);

			cvtColor(src, src_select, CV_BGR2GRAY);
			cvtColor(dst, dst_select, CV_BGR2GRAY);
			GaussianBlur(src_select, src_select, Size(3, 3), 0, 0);
			GaussianBlur(dst_select, dst_select, Size(3, 3), 0, 0);
			GaussianBlur(src, src, Size(3, 3), 0, 0);
			GaussianBlur(dst, dst, Size(3, 3), 0, 0);
			double src_mean = compute_mean(src_select);
			double dst_mean = compute_mean(dst_select);
			cout << "src_mean=" << src_mean << endl;;
			cout << "dst_mean=" << dst_mean << endl;
			/////////////////////////////»ñÈ¡¶þÖµ»¯ºóµÄÐ¡¾ØÐÎ///////////////////////////
			vector<KeyPoint>kp1, kp2;

			int maxcorners = 5000;
			double qualitylevel = 0.01;
			double mindistance = 5;
			Mat mask1, mask2;
			GoodFeaturesToTrackDetector detector(maxcorners, qualitylevel, mindistance);
			progress = 0.2;

			m_RecInfoCall(30 + 33 / warp_H.size() * warp_i + 33 / warp_H.size() * 2 / 6);
			detector.detect(src_select, kp1, mask1);
			detector.detect(dst_select, kp2, mask2);

			FREAK descriptor_extractor;

			cout << "µÚÒ»·ùÍ¼µãÊý=" << kp1.size() << endl;
			cout << "µÚ¶þ·ùÍ¼µãÊý=" << kp2.size() << endl;
			//Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create("SIFT");//´´½¨ÌØÕ÷ÏòÁ¿Éú³ÉÆ÷   
			Mat descriptor1, descriptor2;
			descriptor_extractor.compute(src_select, kp1, descriptor1);
			descriptor_extractor.compute(dst_select, kp2, descriptor2);
			vector<DMatch> matches;
			BruteForceMatcher<HammingLUT>matcher;
			matcher.match(descriptor1, descriptor2, matches);


			cout << "×¼È·µÄµãÊý=" << matches.size() << endl;
			Mat img_matches;
			drawMatches(src_select, kp1, dst_select, kp2, matches, img_matches);
			Mat img_matches_resize;
			resize(img_matches, img_matches_resize, Size(1600, 900), 0, 0, CV_INTER_LINEAR);
			//imshow("matches", img_matches_resize);
			//cvWaitKey(0);
			progress = 0.3;

			m_RecInfoCall(30 + 33 / warp_H.size() * warp_i + 33 / warp_H.size() * 3 / 6);
			vector<Point2d>point_src, point_dst;
			for (int i = 0; i < matches.size(); i++)
			{
				Point2d pt_temp_train, pt_temp_query;
				pt_temp_query.x = kp1[matches[i].queryIdx].pt.x;
				pt_temp_query.y = kp1[matches[i].queryIdx].pt.y;
				pt_temp_train.x = kp2[matches[i].trainIdx].pt.x;
				pt_temp_train.y = kp2[matches[i].trainIdx].pt.y;
				point_src.push_back(pt_temp_query);
				point_dst.push_back(pt_temp_train);
			}
			progress = 0.3;
			src_result.setTo(0);
			//2ÖÐµÄµã=t*1ÖÐµÄµã;Ò²¾ÍÊÇ½«1±ä»¯µ½2ÖÐ
			//if (point_src.size()>20)
			//{
			//Mat T = Homography_Nelder_Mead_with_outliers(point_src, point_dst, 5000);
			Mat T = findHomography(point_src, point_dst, CV_RANSAC);
			cout << T << endl;
			progress = 0.4;
			Mat point1(3, 1, CV_64FC1);


			m_RecInfoCall(30 + 33 / warp_H.size() * warp_i + 33 / warp_H.size() * 4 / 6);
			int block_h = src.size().height / 50;// int(double(MIN(src.size().height, src.size().width)) / 1000 * 30);
			int block_w = src.size().width / 50;
			for (int i = 0; i <= src.size().height - block_h; i = i + block_h)
				for (int j = 0; j <= src.size().width - block_w; j = j + block_w)
					//for (int i = 0; i < boundRect_src.size(); i++)
				{
					//cout << i << endl;
					double x = j;
					double y = i;
					double w = block_w;
					double h = block_h;

					Mat src_temp = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
					double centerx = x + w / 2;
					double centery = y + h / 2;
					int index_min = -1;
					double min = INT_MAX;

					point1.row(0).col(0) = x;
					point1.row(1).col(0) = y;
					point1.row(2).col(0) = 1.0;
					Mat point2;
					//if (index_min>=0)
					//Mat	point2 = TT[index_min] * point1;
					//else
					//point2 = T * point1;
					point2 = cal_cor(T, point1);
					double x2 = MIN(MAX(point2.at<double>(0, 0), 0), dst.size().width - w);
					double y2 = MIN(MAX(point2.at<double>(1, 0), 0), dst.size().height - h);
					int drift_x = 3; //int(MAX(3, MIN(0.3*w, 5)));
					int drift_y = 3;// int(MAX(3, MIN(0.3*h, 5)));
					//Ìí¼ÓÅÐ¶Ï//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					//cout << "x1===" << x << " y1===" << y << endl;
					//cout << "x2===" << x2 << " y2===" << y2 << endl;

					if (x2 >= 0 && y2 >= 0 && x2 + w <= dst.size().width && y2 + h <= dst.size().height)
					{
						vector<double>gray_diff_min;
						vector<int>gray_diff_xx2;
						vector<int>gray_diff_yy2;
						for (int xx2 = x2 - drift_x; xx2 <= x2 + drift_x; xx2 = xx2 + 1)
							for (int yy2 = y2 - drift_y; yy2 <= y2 + drift_y; yy2 = yy2 + 1)
							{
								if (xx2 >= 0 && yy2 >= 0 && xx2 + w <= src.size().width && yy2 + h <= src.size().height)
								{
									Mat dst_temp = dst_select.rowRange(max_my(yy2, 0), max_my(yy2, 0) + h).colRange(max_my(xx2, 0), max_my(xx2, 0) + w).clone();

									gray_diff_min.push_back(sum_diff(src_temp, dst_temp));
									gray_diff_xx2.push_back(xx2);
									gray_diff_yy2.push_back(yy2);
								}
							}
						int index_min = calculate_min(gray_diff_min);
						if (index_min != -1)
						{
							gray_diff.push_back(gray_diff_min[index_min]);
							//cocy_rgb(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src, dst);
							copy_im(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src_select, dst_select);
						}
					}
				}
			//×îºóÒ»ÐÐ
			m_RecInfoCall(30 + 33 / warp_H.size() * warp_i + 33 / warp_H.size() * 5 / 6);
			for (int j = 0; j <= src.size().width - block_w; j = j + block_w)
				//for (int i = 0; i < boundRect_src.size(); i++)
			{
				//cout << i << endl;
				double x = j;
				double y = src.size().height - block_h;
				double w = block_w;
				double h = block_h;

				Mat src_temp = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
				double centerx = x + w / 2;
				double centery = y + h / 2;
				int index_min = -1;
				double min = INT_MAX;

				point1.row(0).col(0) = x;
				point1.row(1).col(0) = y;
				point1.row(2).col(0) = 1.0;
				Mat point2;
				//if (index_min>=0)
				//Mat	point2 = TT[index_min] * point1;
				//else
				//point2 = T * point1;
				point2 = cal_cor(T, point1);
				double x2 = MIN(MAX(point2.at<double>(0, 0), 0), dst.size().width - w);
				double y2 = MIN(MAX(point2.at<double>(1, 0), 0), dst.size().height - h);
				int drift_x = 3; //int(MAX(3, MIN(0.3*w, 5)));
				int drift_y = 3;// int(MAX(3, MIN(0.3*h, 5)));
				//Ìí¼ÓÅÐ¶Ï//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//cout << "x1===" << x << " y1===" << y << endl;
				//cout << "x2===" << x2 << " y2===" << y2 << endl;

				if (x2 >= 0 && y2 >= 0 && x2 + w <= dst.size().width && y2 + h <= dst.size().height)
				{
					vector<double>gray_diff_min;
					vector<int>gray_diff_xx2;
					vector<int>gray_diff_yy2;
					for (int xx2 = x2 - drift_x; xx2 <= x2 + drift_x; xx2 = xx2 + 1)
						for (int yy2 = y2 - drift_y; yy2 <= y2 + drift_y; yy2 = yy2 + 1)
						{
							if (xx2 >= 0 && yy2 >= 0 && xx2 + w <= src.size().width && yy2 + h <= src.size().height)
							{
								Mat dst_temp = dst_select.rowRange(max_my(yy2, 0), max_my(yy2, 0) + h).colRange(max_my(xx2, 0), max_my(xx2, 0) + w).clone();

								gray_diff_min.push_back(sum_diff(src_temp, dst_temp));
								gray_diff_xx2.push_back(xx2);
								gray_diff_yy2.push_back(yy2);
							}
						}
					int index_min = calculate_min(gray_diff_min);
					if (index_min != -1)
					{
						gray_diff.push_back(gray_diff_min[index_min]);
						//cocy_rgb(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src, dst);
						copy_im(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src_select, dst_select);
					}
				}
			}
			m_RecInfoCall(30 + 33 / warp_H.size() * warp_i + 33 / warp_H.size() * 6 / 6);
			//×îºóÒ»ÁÐ
			for (int i = 0; i <= src.size().height - block_h; i = i + block_h)
				//for (int i = 0; i < boundRect_src.size(); i++)
			{
				//cout << i << endl;
				double x = src.size().width - block_w;
				double y = i;
				double w = block_w;
				double h = block_h;

				Mat src_temp = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
				double centerx = x + w / 2;
				double centery = y + h / 2;
				int index_min = -1;
				double min = INT_MAX;

				point1.row(0).col(0) = x;
				point1.row(1).col(0) = y;
				point1.row(2).col(0) = 1.0;
				Mat point2;
				//if (index_min>=0)
				//Mat	point2 = TT[index_min] * point1;
				//else
				//point2 = T * point1;
				point2 = cal_cor(T, point1);
				double x2 = MIN(MAX(point2.at<double>(0, 0), 0), dst.size().width - w);
				double y2 = MIN(MAX(point2.at<double>(1, 0), 0), dst.size().height - h);
				int drift_x = 3; //int(MAX(3, MIN(0.3*w, 5)));
				int drift_y = 3;// int(MAX(3, MIN(0.3*h, 5)));
				//Ìí¼ÓÅÐ¶Ï//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				//cout << "x1===" << x << " y1===" << y << endl;
				//cout << "x2===" << x2 << " y2===" << y2 << endl;

				if (x2 >= 0 && y2 >= 0 && x2 + w <= dst.size().width && y2 + h <= dst.size().height)
				{
					vector<double>gray_diff_min;
					vector<int>gray_diff_xx2;
					vector<int>gray_diff_yy2;
					for (int xx2 = x2 - drift_x; xx2 <= x2 + drift_x; xx2 = xx2 + 1)
						for (int yy2 = y2 - drift_y; yy2 <= y2 + drift_y; yy2 = yy2 + 1)
						{
							if (xx2 >= 0 && yy2 >= 0 && xx2 + w <= src.size().width && yy2 + h <= src.size().height)
							{
								Mat dst_temp = dst_select.rowRange(max_my(yy2, 0), max_my(yy2, 0) + h).colRange(max_my(xx2, 0), max_my(xx2, 0) + w).clone();

								gray_diff_min.push_back(sum_diff(src_temp, dst_temp));
								gray_diff_xx2.push_back(xx2);
								gray_diff_yy2.push_back(yy2);
							}
						}
					int index_min = calculate_min(gray_diff_min);
					if (index_min != -1)
					{
						gray_diff.push_back(gray_diff_min[index_min]);
						//cocy_rgb(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src, dst);
						copy_im(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src_select, dst_select);
					}
				}
			}

			double x = src.size().width - block_w;
			double y = src.size().height - block_h;
			double w = block_w;
			double h = block_h;

			Mat src_temp = src_select.rowRange(y, y + h).colRange(x, x + w).clone();
			double centerx = x + w / 2;
			double centery = y + h / 2;
			int index_min = -1;
			double min = INT_MAX;

			point1.row(0).col(0) = x;
			point1.row(1).col(0) = y;
			point1.row(2).col(0) = 1.0;
			Mat point2;
			//if (index_min>=0)
			//Mat	point2 = TT[index_min] * point1;
			//else
			//point2 = T * point1;
			point2 = cal_cor(T, point1);
			double x2 = MIN(MAX(point2.at<double>(0, 0), 0), dst.size().width - w);
			double y2 = MIN(MAX(point2.at<double>(1, 0), 0), dst.size().height - h);
			int drift_x = 3; //int(MAX(3, MIN(0.3*w, 5)));
			int drift_y = 3;// int(MAX(3, MIN(0.3*h, 5)));
			//Ìí¼ÓÅÐ¶Ï//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//cout << "x1===" << x << " y1===" << y << endl;
			//cout << "x2===" << x2 << " y2===" << y2 << endl;

			if (x2 >= 0 && y2 >= 0 && x2 + w <= dst.size().width && y2 + h <= dst.size().height)
			{
				vector<double>gray_diff_min;
				vector<int>gray_diff_xx2;
				vector<int>gray_diff_yy2;
				for (int xx2 = x2 - drift_x; xx2 <= x2 + drift_x; xx2 = xx2 + 1)
					for (int yy2 = y2 - drift_y; yy2 <= y2 + drift_y; yy2 = yy2 + 1)
					{
						if (xx2 >= 0 && yy2 >= 0 && xx2 + w <= src.size().width && yy2 + h <= src.size().height)
						{
							Mat dst_temp = dst_select.rowRange(max_my(yy2, 0), max_my(yy2, 0) +

								h).colRange(max_my(xx2, 0), max_my(xx2, 0) + w).clone();

							gray_diff_min.push_back(sum_diff(src_temp, dst_temp));
							gray_diff_xx2.push_back(xx2);
							gray_diff_yy2.push_back(yy2);
						}
					}
				int index_min = calculate_min(gray_diff_min);
				if (index_min != -1)
				{
					gray_diff.push_back(gray_diff_min[index_min]);
					//cocy_rgb(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src, dst);
					copy_im(src_result, x, y, h, w, gray_diff_xx2[index_min], gray_diff_yy2[index_min], src_select, dst_select);
				}
			}
			stringstream sss;
			sss << warp_i;
			imwrite(p + "result_base" + sss.str() + ".jpg", src_result);
			//double thres_src = getThreshVal_Otsu_8u(src_result);
		}
	}
	m_RecInfoCall(65);
	return 0;
}

