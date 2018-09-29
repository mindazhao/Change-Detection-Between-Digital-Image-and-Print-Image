#include"iostream"
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
#include<ctime>
#include<random>
#define max_my(a, b)  (((a) > (b)) ? (a) : (b))
#define     NO_OBJECT       0  
#define     MIN(x, y)       (((x) < (y)) ? (x) : (y))  
#define     ELEM(img, r, c) (CV_IMAGE_ELEM(img, unsigned char, r, c))  
#define     ONETWO(L, r, c, col) (L[(r) * (col) + c])  

using namespace std;
using namespace cv;

struct cor
{
	int x;
	int y;
};
cor find_start(vector<vector<int>>flag, vector<vector<int>>process)
{
	cor temp;
	temp.x = -1; temp.y = -1;
	for (int i = 0; i < flag.size(); i++)
		for (int j = 0; j < flag[i].size(); j++)
		{
			if (flag[i][j] == 1 && process[i][j] == 0)
			{

				temp.x = i, temp.y = j;
				return temp;
			}

		}

	return temp;
}

vector<cor> fuse(cor temp, vector<vector<int>>&flag, vector<vector<int>>&process, Mat &middle, int width_block, int height_block, vector<cor>&stack, double fuse_threshold, double &area, int level)
{
	process[temp.x][temp.y] = level;
	for (int i = -1; i < 2; i = i + 1)
		for (int j = -1; j < 2; j = j + 1)
		{
			if (temp.x + i<flag.size() && temp.x + i >= 0 && temp.y + j<flag[0].size() && temp.y + j >= 0)
				if (flag[temp.x + i][temp.y + j] == 1 && process[temp.x + i][temp.y + j] == 0)
				{
					rectangle(middle, Point(width_block*(temp.x + i), height_block*(temp.y + j)), Point(width_block*(temp.x + i + 1), height_block*(temp.y + j + 1)), Scalar(255, 255, 255), -1, 8);

					std::vector<std::vector<cv::Point>> contours;
					cv::findContours(middle,
						contours, // a vector of contours   
						CV_RETR_TREE,//CV_RETR_EXTERNAL,//CV_RETR_TREE, // retrieve the external contours  
						CV_LINK_RUNS);//CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours  
					double g_dConArea = contourArea(contours[0], true);
					RotatedRect rect = minAreaRect(contours[0]);
					//cout << rect.size.area << endl;
					if (stack.size()*width_block*height_block < fuse_threshold*rect.size.width*rect.size.height)
						return stack;
					else
					{
						cor tp;
						tp.x = temp.x + i; tp.y = temp.y + j;
						stack.push_back(tp);
						process[temp.x + i][temp.y + j] = level;
						area += 1;
						fuse(tp, flag, process, middle, width_block, height_block, stack, fuse_threshold, area, level);

					}
				}
		}
	return stack;
}

void fuse_process(Mat I, int erode_threshold, int size_block_width, int size_block_height, std::vector<std::vector<cv::Point>> &contours_whole, vector<double>&area, double fuse_threshold, vector<vector<int>>&process, int level, vector<int>&value)
{
	Mat I_color = I.clone();
	I_color.setTo(255);
	cvtColor(I, I, CV_BGR2GRAY);
	Mat kern = getStructuringElement(MORPH_RECT, Size(erode_threshold, erode_threshold));
	erode(I, I, kern);

	Mat result = I.clone();
	result.setTo(0);

	int width_block = ceil(double(I.size().width) / size_block_width);
	int height_block = ceil(double(I.size().height) / size_block_height);
	vector<vector<int>>flag;
	flag.resize(size_block_width);
	for (int i = 0; i < size_block_width; i++)
	{
		flag[i].resize(size_block_height);
	}
	for (int i = 0; i < I.size().width - width_block; i = i + width_block)
		for (int j = 0; j < I.size().height - height_block; j = j + height_block)
		{
			Mat dst_temp = I.rowRange(j, j + height_block).colRange(i, i + width_block).clone();
			Mat dst_temp_binary;
			threshold(dst_temp, dst_temp_binary, 100, 255, CV_THRESH_BINARY);
			int count = countNonZero(dst_temp_binary);

			if (count>5)
			{
				flag[i / width_block][j / height_block] = 1;
				//rectangle(result, Point(i, j), Point(i + width_block, j + height_block), Scalar(255, 255, 255), -1, 8);
			}
			else
				flag[i / width_block][j / height_block] = 0;

		}


	cor start = find_start(flag, process);
	int i = 0;

	std::vector<std::vector<cv::Point>> contours;
	while (start.x != -1)
	{
		i++;

		vector<cor>stack;
		stack.push_back(start);
		double area_temp = 1;
		Mat result_temp = result.clone();
		rectangle(result_temp, Point(width_block*(start.x), height_block*(start.y)), Point(width_block*(start.x + 1), height_block*(start.y + 1)), Scalar(255, 255, 255), -1, 8);
		vector<cor>re = fuse(start, flag, process, result_temp, width_block, height_block, stack, fuse_threshold, area_temp, level);
		result_temp.setTo(0);
		for (int i = 0; i < re.size(); i++)
		{
			rectangle(result_temp, Point(width_block*(re[i].x), height_block*(re[i].y)), Point(width_block*(re[i].x + 1), height_block*(re[i].y + 1)), Scalar(255, 255, 255), -1, 8);
		}


		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(result_temp,
			contours, // a vector of contours   
			CV_RETR_EXTERNAL,//CV_RETR_EXTERNAL,//CV_RETR_TREE, // retrieve the external contours  
			CV_LINK_RUNS);//CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours  
		if (area_temp<10000)
			drawContours(I_color, contours, -1, Scalar(255 * (i % 2), 255 * ((i + 1) % 2), 255 * (i % 2)), 3);
		//绘制最小外接矩形


		area.push_back(area_temp);
		value.push_back(level + 9);
		contours_whole.push_back(contours[0]);

		//rectangle(I_color, Point(rect_temp.x, rect_temp.y), Point(rect_temp.x + rect_temp.width, rect_temp.y + rect_temp.height), Scalar(0, 0, 0), 2, 8);
		cout << "i==" << i << "   " << area_temp << endl;
		start = find_start(flag, process);
	}
	stringstream ss;
	string str;
	ss << level;
	ss >> str;

	//imwrite("result_fuse" + str + ".jpg", I_color);
}

static std::vector<std::vector<cv::Point>> find1(Mat image)
{
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(image,
		contours, // a vector of contours   
		CV_RETR_TREE, // retrieve the external contours  
		CV_CHAIN_APPROX_NONE); // retrieve all pixels of each contours  

	return contours;
}

Mat cal_cor(Mat H, Mat point1)
{
	Mat point2(3, 1, CV_64FC1);
	point2.row(0).col(0) = int((H.at<double>(0, 0)*(point1.at<double>(0, 0)) + H.at<double>(0, 1) * point1.at<double>(0, 1) + H.at<double>(0, 2)) / (H.at<double>(2, 0)*point1.at<double>(0, 0) + H.at<double>(2, 1) *point1.at<double>(1, 0) + H.at<double>(2, 2)));
	point2.row(1).col(0) = int((H.at<double>(1, 0)*(point1.at<double>(0, 0)) + H.at<double>(1, 1) * point1.at<double>(0, 1) + H.at<double>(1, 2)) / (H.at<double>(2, 0)*point1.at<double>(0, 0) + H.at<double>(2, 1) *point1.at<double>(1, 0) + H.at<double>(2, 2)));
	point2.row(2).col(0) = 1;
	return point2;
}




extern "C" __declspec(dllexport) int __stdcall textcombine(const char* path, const char* matrix, int level, int ba, int color,int thres,int erode)
{
	
//int main(){
	Mat src_1;//黑白图

	//string path = "F:\\developfiles\\DifferentialDetection\\DifferentialDetection\\bin\\x64\\Release\\templates\\2017101806\\detections\\20\\";
	//int level = 15;
	//int color = 1;
	//int ba = 0;

	vector<Mat>warp_H;
	string mats = matrix;
	istringstream iss(mats);
	string sub;
	m_RecInfoCall(50);
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
	bool based = false;
	bool colored = false;
	//cout << "1" << endl;
	if (color == 1)
		colored = true;
	if (ba == 1)
		based = true;
	cout << path << endl;
	string p = path;
	
	ofstream myfile_contours_multi((p + "contours_multi.result").c_str(), ios::out);

	int length = level;
	//int thred = (src_whole.cols / 2 + src_whole.rows / 2) / 8;//直线检测的长度阈值
	
	for (int warp_i = 0; warp_i < warp_H.size(); warp_i++){
		m_RecInfoCall(0 + 99 / warp_H.size() * warp_i + 99 / warp_H.size() * 1 / 4);
		Mat src2;
		Mat src_color;
		Mat src_color2;
		int r;
		int c;
		stringstream ss;
		ss << warp_i;
		ofstream myfile_contours((p + "contours"+ss.str()+".result").c_str(), ios::out);
		if (based){
			src_1 = imread((p + "result_base"+ss.str()+".jpg").c_str());
			cvtColor(src_1, src2, CV_BGR2GRAY);
			r = src2.rows;
			c = src2.cols;

		}
		
		if (colored){
			src_color = imread((p + "result_color" + ss.str() + ".jpg").c_str());
			cvtColor(src_color, src_color2, CV_BGR2GRAY);
			r = src_color2.rows;
			c = src_color2.cols;
		}

		m_RecInfoCall(0 + 99 / warp_H.size() * warp_i + 99 / warp_H.size() * 2 / 4);
		
		Mat a = Mat(r, c, CV_8U);
		for (int i = 0; i < r; i++){
			for (int j = 0; j < c; j++){
				if (based && colored){
					a.row(i).col(j) = max(src2.at<uchar>(i, j), src_color2.at<uchar>(i,j));
				}
				else if (based){
					a.row(i).col(j) = src2.at<uchar>(i, j);
				}
				else if (colored){
					a.row(i).col(j) = src_color2.at<uchar>(i, j);
				}
			}
		}
		imwrite(p + "resultcombine" + ss.str() + ".jpg", a);
		/*imshow("a", a);
		waitKey();*/
		std::vector<std::vector<cv::Point>> contours;
		std::vector<double>area;
		std::vector<int>value;


		
		int erode_threshold = erode;
		int size_block_width = MIN(a.size().width / 30, 100);
		int size_block_height = MIN(a.size().height / 30, 100);
		double fuse_threshold = 0.5;

		vector<vector<int>>process;
		process.resize(size_block_width);
		for (int i = 0; i < size_block_width; i++)
		{
			process[i].resize(size_block_height);
		}
		m_RecInfoCall(0 + 99 / warp_H.size() * warp_i + 99 / warp_H.size() * 3 / 4);
		int level = 100;
		for (level; level >= thres; level = level - 10){
			Mat src_select_binary;
			threshold(a, src_select_binary, level, 255, CV_THRESH_BINARY);
			stringstream sss;
			sss << level;
			imwrite(p + "binary_"+sss.str()+".jpg", src_select_binary);
			Mat pic = imread(p + "binary_" + sss.str() + ".jpg");
			fuse_process(pic, erode_threshold, size_block_width, size_block_height, contours, area, fuse_threshold, process, level, value);
			for (int i = 0; i < contours.size(); i++){
				for (int j = 0; j < contours[i].size(); j++){
					myfile_contours << contours[i][j].x << "," << contours[i][j].y << ",";
					Mat point1(3, 1, CV_64FC1);
					Mat point2;
					point1.row(0).col(0) = contours[i][j].x;
					point1.row(1).col(0) = contours[i][j].y;
					point1.row(2).col(0) = 1.0;
					point2 = cal_cor(warp_H[warp_i], point1);
					myfile_contours_multi << (int)point2.at<double>(0, 0) << "," << (int)point2.at<double>(1, 0) << "," << contours[i][j].x << "," << contours[i][j].y << ",";
				}
				myfile_contours << value[i] << ",";
				myfile_contours << area[i] << ",0" << endl;
				myfile_contours_multi << value[i] << ",";
				myfile_contours_multi << area[i] << ",0" << endl;
			}
			contours.clear();
			area.clear();
			value.clear();
		}
		m_RecInfoCall(0 + 99 / warp_H.size() * warp_i + 99 / warp_H.size() * 4 / 4);
			
		
		myfile_contours.close();
	}
	
	myfile_contours_multi.close();
	//cout << "7" << endl;
	m_RecInfoCall(100);
	return 0;
}
