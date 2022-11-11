/*
reference: https://github.com/ethan-li-coding/PatchMatchStereo
*/

#include "stdafx.h"
#include <iostream>
#include "PatchMatchStereo.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cstdint>

using namespace std;

void writeMatToFile(cv::Mat &m, const char *filename)
{
	ofstream fout(filename);

	if (!fout)
	{
		cout << "File Not Opened" << endl;
		return;
	}

	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			fout << m.at<double>(i, j) << "\t";
		}
		fout << endl;
	}

	fout.close();
}

int main(int argc, char *argv[])
{
	std::vector<std::string> args;
	args.assign(argv + 1, argv + argc);

	std::string img_path_left = args[0];
	std::string img_path_right = args[1];
	std::string disp_path_left = args[2];
	std::string disp_path_right = args[3];
	std::string plane_path_left = args[4];
	std::string plane_path_right = args[5];
	std::string edge_path_left = args[6];
	std::string edge_path_right = args[7];

	int patch_size = std::stoi(args[8]);
	int num_iters = std::stoi(args[9]);
	std::string output_path = args[10];

	cv::Mat img_left = cv::imread(img_path_left, 0);
	cv::Mat img_right = cv::imread(img_path_right, 0);

	if (img_left.data == nullptr || img_right.data == nullptr)
	{
		std::cout << "Fail to read images！" << std::endl;
		return -1;
	}
	cv::Mat disp_left = cv::imread(disp_path_left, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
	cv::Mat disp_right = cv::imread(disp_path_right, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

	cv::Mat plane_left = cv::imread(plane_path_left, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
	cv::Mat plane_right = cv::imread(plane_path_right, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

	cv::Mat edge_left = cv::imread(edge_path_left, 0);
	cv::Mat edge_right = cv::imread(edge_path_right, 0);

	// data type check
	if (!(disp_left.type() == CV_16UC1 && disp_right.type() == CV_16UC1))
	{
		std::cout << "Disp left and Disp right should be CV_16UC1" << std::endl;
		return -1;
	}
	if (!(plane_left.type() == CV_16UC3 && plane_right.type() == CV_16UC3))
	{
		std::cout << "Plane left and Plane right should be CV_16UC3" << std::endl;
		return -1;
	}

	const sint32 width = static_cast<uint32>(img_left.cols);
	const sint32 height = static_cast<uint32>(img_right.rows);

	// PMS匹配参数设计
	PMSOption pms_option;
	// patch大小
	pms_option.patch_size = patch_size;
	// 候选视差范围
	pms_option.min_disparity = 12;
	pms_option.max_disparity = 96;
	// gamma
	pms_option.gamma = 10.0f;
	// alpha
	pms_option.alpha = 0.9f;
	// t_col
	pms_option.tau_col = 10.0f;
	// t_grad
	pms_option.tau_grad = 2.0f;
	// 传播迭代次数
	pms_option.num_iters = num_iters;

	// 一致性检查
	pms_option.is_check_lr = true;
	pms_option.lrcheck_thres = 1.0f;
	// 视差图填充
	pms_option.is_fill_holes = true;

	// 前端平行窗口
	pms_option.is_fource_fpw = false;

	// 整数视差精度
	pms_option.is_integer_disp = false;

	// 定义PMS匹配类实例
	PatchMatchStereo pms;
	pms.Initialize(width, height, pms_option,
				   img_left, img_right, disp_left, disp_right,
				   plane_left, plane_right, edge_left, edge_right);

	auto disp_prop_left = new float32[uint32(width * height)]();
	pms.Propogate(disp_prop_left);
	cv::Mat mat_disp_prop(height, width, CV_64FC1);

	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			mat_disp_prop.at<double>(i, j) = disp_prop[i * width + j]
		}
	}
	writeMatToFile(mat_disp_prop, output_path.c_str());
}