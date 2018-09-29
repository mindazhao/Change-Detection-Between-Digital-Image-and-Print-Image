#include "stdafx.h"
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

using namespace std;
using namespace cv;
double pi = 3.1415926;


class location
{
public:
	int x;
	int y;
	int width;
	int height;
};
Mat src;
Mat dst;

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

Mat cal_cor(Mat H, Mat point1)
{
	Mat point2(3, 1, CV_64FC1);
	point2.row(0).col(0) = int((H.at<double>(0, 0)*(point1.at<double>(0, 0)) + H.at<double>(0, 1) * point1.at<double>(0, 1) + H.at<double>(0, 2)) / (H.at<double>(2, 0)*point1.at<double>(0, 0) + H.at<double>(2, 1) *point1.at<double>(1, 0) + H.at<double>(2, 2)));
	point2.row(1).col(0) = int((H.at<double>(1, 0)*(point1.at<double>(0, 0)) + H.at<double>(1, 1) * point1.at<double>(0, 1) + H.at<double>(1, 2)) / (H.at<double>(2, 0)*point1.at<double>(0, 0) + H.at<double>(2, 1) *point1.at<double>(1, 0) + H.at<double>(2, 2)));
	point2.row(2).col(0) = 1;
	return point2;
}


template < typename T>
vector< size_t>  sort_indexes(const vector< T>  & v) {

	// initialize original index locations
	vector< size_t>  idx(v.size());
	for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] <  v[i2]; });

	return idx;
}

extern "C" __declspec(dllexport) int compare(char* src_str, char* dst_str, int  clusterCount, char* path, int block_w, int block_h)
{

	




	Ptr<DescriptorMatcher> descriptor_matcher;
	string detectorType = "SIFT";
	string descriptorType = "SIFT";
	string matcherType = "FlannBased";

	Ptr<FeatureDetector> featureDetector;
	Ptr<DescriptorExtractor> descriptorExtractor;
	if (!createDetectorDescriptorMatcher(detectorType, descriptorType, matcherType, featureDetector, descriptorExtractor, descriptor_matcher))
	{
		cout << "Creat Detector Descriptor Matcher False!" << endl;
		return -1;
	}  // = DescriptorMatcher;//::create("BruteForce");
	string s = src_str;
	string d = dst_str;
	string p = path;
	src = imread(s);
	dst = imread(d);

	Mat dst_resize, src_resize;
	//20171030 new-add
	double resize_rate = 1;
	int change_flag = 0;
	if (dst.size().width > 5000 || dst.size().height > 5000)
	{
		change_flag = 1;
		resize_rate = 5000 / double(MAX(dst.size().height, dst.size().width));
		resize(dst, dst_resize, Size(resize_rate*dst.size().width, resize_rate*dst.size().height), 0, 0, INTER_LINEAR);
		resize(src, src_resize, Size(resize_rate*src.size().width, resize_rate*src.size().height), 0, 0, INTER_LINEAR);

	}
	
	if (src_resize.size().width > 5000 || src_resize.size().height > 5000)
	{
		resize_rate = resize_rate * 5000 / double(MAX(src_resize.size().height, src_resize.size().width));
		resize(dst, dst_resize, Size(resize_rate*dst.size().width, resize_rate*dst.size().height), 0, 0, INTER_LINEAR);
		resize(src, src_resize, Size(resize_rate*src.size().width, resize_rate*src.size().height), 0, 0, INTER_LINEAR);
	}

	imwrite(p + "src_resize.jpg", src_resize);
	imwrite(p + "dst_resize.jpg", dst_resize);
	vector<KeyPoint>kp1, kp2;
	
	int sift_num = 0;
	if (clusterCount == 1)
		sift_num = 5000;
	else
		sift_num = max(5000 / clusterCount, 500 * clusterCount);
	if (change_flag == 1)
	{
		SiftFeatureDetector  siftdtc_img1(sift_num, 3, 0.01, 10, 1.6);
		siftdtc_img1.detect(src_resize, kp1);
		//cout << "kp1 size:" << kp1.size() << endl;
		SiftFeatureDetector siftdtc_img2(kp1.size()*clusterCount, 3, 0.01, 10, 1.6);
		siftdtc_img2.detect(dst_resize, kp2);

	}
	else
	{
		SiftFeatureDetector  siftdtc_img1(sift_num, 3, 0.01, 10, 1.6);
		siftdtc_img1.detect(src, kp1);
		//cout << "kp1 size:" << kp1.size() << endl;
		SiftFeatureDetector siftdtc_img2(kp1.size()*clusterCount, 3, 0.01, 10, 1.6);
		siftdtc_img2.detect(dst, kp2);
	}
	

	m_RecInfoCall(5);
	Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create("SIFT"); 
	Mat descriptor1, descriptor2;
	
	if (change_flag == 0)
	{
		descriptor_extractor->compute(src, kp1, descriptor1);
		descriptor_extractor->compute(dst, kp2, descriptor2);
	}
	else
	{
		descriptor_extractor->compute(src_resize, kp1, descriptor1);
		descriptor_extractor->compute(dst_resize, kp2, descriptor2);
	}


	vector<vector<DMatch>> knnMatches;
	vector<DMatch> matches;
	descriptor_matcher->knnMatch(descriptor2, descriptor1, knnMatches, 2);
	const float minRatio = 1.f / 1.5f;
	int num = 0;
	m_RecInfoCall(10);
	
	for (size_t i = 0; i < knnMatches.size(); i++)
	{
		cv::DMatch& bestMatch = knnMatches[i][0];
		cv::DMatch& betterMatch = knnMatches[i][1];
		float distanceRatio = bestMatch.distance / betterMatch.distance;
		if (distanceRatio < minRatio)
		{
			matches.push_back(bestMatch);
			num++;

		}
	}

	cout << "match_size=" << matches.size() << endl;
	Mat img_matches;
	//drawMatches(dst, kp2, src, kp1, matches, img_matches);
	Mat img_matches_resize;
	//resize(img_matches, img_matches_resize, Size(1600, 900), 0, 0, CV_INTER_LINEAR);
	//imshow("matches", img_matches_resize);
	//cvWaitKey(0);
	std::vector<vector<Point2f>> kmeans_result_query, kmeans_result_train;
	kmeans_result_query.resize(clusterCount);
	kmeans_result_train.resize(clusterCount);
	Vector<Mat>H_sum;
	H_sum.resize(clusterCount);
	vector<CvRect>rect_sum;
	rect_sum.resize(clusterCount);
	Mat point1(3, 1, CV_64FC1);
	Mat point2(3, 1, CV_64FC1);
	Mat point3(3, 1, CV_64FC1);
	Mat point4(3, 1, CV_64FC1);
	point1.row(0).col(0) = 0.0;
	point1.row(1).col(0) = 0.0;
	point1.row(2).col(0) = 1.0;
	point2.row(0).col(0) = src.cols - 1;
	point2.row(1).col(0) = src.rows - 1;
	point2.row(2).col(0) = 1.0;

	point3.row(0).col(0) = src.cols - 1;
	point3.row(1).col(0) = 0.0;
	point3.row(2).col(0) = 1.0;

	point4.row(0).col(0) = 0.0;
	point4.row(1).col(0) = src.rows - 1;
	point4.row(2).col(0) = 1.0;

	m_RecInfoCall(15);

	for (int i = 0; i < kp1.size(); i++)
	{
		kp1[i].pt.x = kp1[i].pt.x / resize_rate;
		kp1[i].pt.y = kp1[i].pt.y / resize_rate;
	}

	for (int i = 0; i < kp2.size(); i++)
	{
		kp2[i].pt.x = kp2[i].pt.x / resize_rate;
		kp2[i].pt.y = kp2[i].pt.y / resize_rate;
	}

	ofstream re_change((p + "re_change.txt").c_str(), ios::app | ios::out);
	re_change << kp1.size() << "         " << kp2.size();
	re_change.close();



	

	if (block_w != -1 && block_h != -1)
	{

		for (int p = 0; p < block_w; p++)
			for (int q = 0; q < block_h; q++)
			{
				for (int m = 0; m < matches.size(); m++)
					if (kp2[matches[m].queryIdx].pt.x>p*dst.size().width / block_w&&kp2[matches[m].queryIdx].pt.x<(p + 1)*dst.size().width / block_w&&
						kp2[matches[m].queryIdx].pt.y>q*dst.size().height / block_h&&kp2[matches[m].queryIdx].pt.y < (q + 1)*dst.size().height / block_h)
					{
						Point2f pt_temp_train, pt_temp_query;
						pt_temp_query.x = kp2[matches[m].queryIdx].pt.x;
						pt_temp_query.y = kp2[matches[m].queryIdx].pt.y;
						pt_temp_train.x = kp1[matches[m].trainIdx].pt.x;
						pt_temp_train.y = kp1[matches[m].trainIdx].pt.y;
						kmeans_result_query[q*block_w + p].push_back(pt_temp_query);
						kmeans_result_train[q*block_w + p].push_back(pt_temp_train);
					}
			}
	}
	else
	{
		Mat data(matches.size(), 2, CV_32FC1);
		Mat labels;

		Mat centers(clusterCount, 1, data.type());
		/*float *x = new float[kp2.size()];
		float *y = new float[kp2.size()];
		float *s = new float[kp2.size()];
		float *angle = new float[kp2.size()];*/
		double ang = 0;
		double len = 0;
		double tann = 0;
		double tann_final = 0;
		double kp1_x = 0;
		double kp1_y = 0;
		double center_x = double(src.size().width) / 2;
		double center_y = double(src.size().height) / 2;

		int parameter_kernel = 0;
		

		parameter_kernel = 180;
		for (int i = 0; i < matches.size(); i++)
		{

			kp1_x = kp1[matches[i].trainIdx].pt.x - center_x;
			kp1_y = kp1[matches[i].trainIdx].pt.y - center_y;
			tann = atan(kp1_y / kp1_x) / pi * parameter_kernel;
			if (kp1_x < 0)
				tann = tann + parameter_kernel;

			ang = kp2[matches[i].queryIdx].angle - kp1[matches[i].trainIdx].angle;
			len = sqrt(pow(kp1_x, 2) + pow(kp1_y, 2));
			//tann = atan(double(kp1[matches[i].trainIdx].pt.y) / double(kp1[matches[i].trainIdx].pt.x))/pi*180;
			tann_final = tann + ang;
			kp1_x = len*cos(tann_final / parameter_kernel * pi);
			kp1_y = len*sin(tann_final / parameter_kernel * pi);
			/* kp1_x = kp1[matches[i].trainIdx].pt.x;
			kp1_y = kp1[matches[i].trainIdx].pt.y;*/

			data.row(i).col(0) = kp2[matches[i].queryIdx].pt.x - kp1_x;
			data.row(i).col(1) = kp2[matches[i].queryIdx].pt.y - kp1_y;
			// data.row(i).col(2) = log(kp2[i].size / kp1[matches[i].trainIdx].size);
			//data.row(i).col(2) = kp2[i].angle - kp1[matches[i].trainIdx].angle;
		}
		//ofstream f("result.txt");
		//f << data;
		//f.close();

		kmeans(data, clusterCount, labels,
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
			3, KMEANS_PP_CENTERS, centers);

		for (int i = 0; i < matches.size(); i++)  
		{
			Point2f pt_temp_train, pt_temp_query;
			pt_temp_query.x = kp2[matches[i].queryIdx].pt.x;
			pt_temp_query.y = kp2[matches[i].queryIdx].pt.y;
			pt_temp_train.x = kp1[matches[i].trainIdx].pt.x;
			pt_temp_train.y = kp1[matches[i].trainIdx].pt.y;
			kmeans_result_query[int(labels.at<int>(i, 0))].push_back(pt_temp_query);
			kmeans_result_train[int(labels.at<int>(i, 0))].push_back(pt_temp_train);
		}

	}
	m_RecInfoCall(20);
	ofstream myfile_location((p + "location.dat").c_str(), ios::app | ios::out);
	ofstream myfile_location_four_vex((p + "location_four_vex.dat").c_str(), ios::app | ios::out);
	ofstream myfile_matrix((p + "matrix.dat").c_str(), ios::app | ios::out);
	for (int i = 0; i < clusterCount; i++)
	{

		if (kmeans_result_query[i].size()>4){
			H_sum[i] = findHomography(kmeans_result_train[i], kmeans_result_query[i], CV_RANSAC);
			//H_sum[i] = H_sum[i].inv();
			cout << H_sum[i] << endl;
			myfile_matrix << H_sum[i].at<double>(0, 0) << " ";
			myfile_matrix << H_sum[i].at<double>(0, 1) << " ";
			myfile_matrix << H_sum[i].at<double>(0, 2) << ",";
			myfile_matrix << H_sum[i].at<double>(1, 0) << " ";
			myfile_matrix << H_sum[i].at<double>(1, 1) << " ";
			myfile_matrix << H_sum[i].at<double>(1, 2) << ",";
			myfile_matrix << H_sum[i].at<double>(2, 0) << " ";
			myfile_matrix << H_sum[i].at<double>(2, 1) << " ";
			myfile_matrix << H_sum[i].at<double>(2, 2) << ";";
			H_sum[i].row(0).col(2) = H_sum[i].row(0).col(2);//*3
			H_sum[i].row(1).col(2) = H_sum[i].row(1).col(2);//
			//Mat point1_perspective = H_sum[i] * point1;
			//Mat point2_perspective = H_sum[i] * point2;
			Mat point1_perspective = cal_cor(H_sum[i], point1);
			Mat point2_perspective = cal_cor(H_sum[i], point2);
			Mat point3_perspective = cal_cor(H_sum[i], point3);
			Mat point4_perspective = cal_cor(H_sum[i], point4);

			CvRect rect_temp;
			//cout << point2_perspective.at<double>(1, 0) << endl;
			if (point1_perspective.at<double>(0, 0) < 0)
				point1_perspective.row(0).col(0) = 0;
			if (point1_perspective.at<double>(1, 0) < 0)
				point1_perspective.row(1).col(0) = 0;
			if (point1_perspective.at<double>(0, 0) >= dst.cols)
				point1_perspective.row(0).col(0) = dst.cols - 1;
			if (point1_perspective.at<double>(1, 0) >= dst.rows)
				point1_perspective.row(1).col(0) = dst.rows - 1;
			if (point2_perspective.at<double>(0, 0) < 0)
				point2_perspective.row(0).col(0) = 0;
			if (point2_perspective.at<double>(1, 0) < 0)
				point2_perspective.row(1).col(0) = 0;
			if (point2_perspective.at<double>(0, 0) >= dst.cols)
				point2_perspective.row(0).col(0) = dst.cols - 1;
			if (point2_perspective.at<double>(1, 0) >= dst.rows)
				point2_perspective.row(1).col(0) = dst.rows - 1;

			if (point3_perspective.at<double>(0, 0) < 0)
				point3_perspective.row(0).col(0) = 0;
			if (point3_perspective.at<double>(1, 0) < 0)
				point3_perspective.row(1).col(0) = 0;
			if (point3_perspective.at<double>(0, 0) >= dst.cols)
				point3_perspective.row(0).col(0) = dst.cols - 1;
			if (point3_perspective.at<double>(1, 0) >= dst.rows)
				point3_perspective.row(1).col(0) = dst.rows - 1;
			if (point4_perspective.at<double>(0, 0) < 0)
				point4_perspective.row(0).col(0) = 0;
			if (point4_perspective.at<double>(1, 0) < 0)
				point4_perspective.row(1).col(0) = 0;
			if (point4_perspective.at<double>(0, 0) >= dst.cols)
				point4_perspective.row(0).col(0) = dst.cols - 1;
			if (point4_perspective.at<double>(1, 0) >= dst.rows)
				point4_perspective.row(1).col(0) = dst.rows - 1;


			//cout << img1.size().height << endl;
			//cout << point2_perspective.at<double>(1, 0) << endl;
			if (point1_perspective.at<double>(0, 0) < point2_perspective.at<double>(0, 0))
				rect_sum[i].x = int(point1_perspective.at<double>(0, 0));
			else
				rect_sum[i].x = int(point2_perspective.at<double>(0, 0));
			rect_sum[i].width = abs(int(point1_perspective.at<double>(0, 0) - point2_perspective.at<double>(0, 0)));
			if (point1_perspective.at<double>(1, 0) < point2_perspective.at<double>(1, 0))
				rect_sum[i].y = int(point1_perspective.at<double>(1, 0));
			else
				rect_sum[i].y = int(point2_perspective.at<double>(1, 0));
			rect_sum[i].height = abs(int(point1_perspective.at<double>(1, 0) - point2_perspective.at<double>(1, 0)));

			myfile_location << rect_sum[i].x << " " << rect_sum[i].y << " " << rect_sum[i].width << " " << rect_sum[i].height << endl;

			myfile_location_four_vex << int(point1_perspective.at<double>(0, 0)) << " " << int(point1_perspective.at<double>(1, 0)) << " "
				<< int(point2_perspective.at<double>(0, 0)) << " " << int(point2_perspective.at<double>(1, 0)) << " "
				<< int(point3_perspective.at<double>(0, 0)) << " " << int(point3_perspective.at<double>(1, 0)) << " "
				<< int(point4_perspective.at<double>(0, 0)) << " " << int(point4_perspective.at<double>(1, 0)) << " " << endl;

			///////////////////////////////////////////////////////////////////////////////////////////////
			Mat imageturn = Mat::zeros(src.size().width, src.size().height, src.type());
			warpPerspective(dst, imageturn, H_sum[i].inv(), cv::Size(imageturn.rows, imageturn.cols));
			//imwrite(p+"nima.jpg", imageturn);
			//cvWaitKey(0);
			CvSize size;
			size.height = src.size().height;
			size.width = src.size().width;
			IplImage *result_src_gray = cvCreateImage(size, 8, 1);
			IplImage *result_dst_gray = cvCreateImage(size, 8, 1);
			cvCvtColor(&IplImage(src), result_src_gray, CV_RGB2GRAY);
			cvCvtColor(&IplImage(imageturn), result_dst_gray, CV_RGB2GRAY);
			/////////////////////////////////////////////////////////////////////////////


		}
	}
	myfile_matrix.close();
	myfile_location.close();
	myfile_location_four_vex.close();

	m_RecInfoCall(30);
	return 0;
}