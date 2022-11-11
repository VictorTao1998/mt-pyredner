/*
reference: https://github.com/ethan-li-coding/PatchMatchStereo
*/

#include "stdafx.h"
#include <iostream>
#include "pms_propagation.h"
#include <fstream>
#include <cstdint>
#include <chrono>

// opencv library
#include <opencv2/opencv.hpp>

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
	std::string conf_path_left = args[3];
	std::string plane_path_left = args[4];
	std::string edge_path_left = args[5];

	int patch_size = std::stoi(args[6]);
	int num_iters = std::stoi(args[7]);
	printf("patch_size: %d, num_iters: %d\n",patch_size, num_iters);
	std::string output_path = args[8];
	auto start = std::chrono::steady_clock::now();
	cv::Mat img_left = cv::imread(img_path_left, 0);
	cv::Mat img_right = cv::imread(img_path_right, 0);

	if (img_left.data == nullptr || img_right.data == nullptr)
	{
		std::cout << "Fail to read images!" << std::endl;
		return -1;
	}
	cv::Mat disp_left = cv::imread(disp_path_left, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
	cv::Mat conf_left = cv::imread(conf_path_left, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
	cv::Mat plane_left = cv::imread(plane_path_left, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
	cv::Mat edge_left = cv::imread(edge_path_left, 0);
	auto end = std::chrono::steady_clock::now();
	auto tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	printf("Image reading: %.3f s\n", tt.count() / 1000000.0);

	// size check
	if (!(disp_left.cols == img_left.cols && disp_left.rows == img_left.rows))
	{
		std::cout << "Disp left and Img Left should have same shape" << std::endl;
		return -1;
	}

	// data type check
	if (!(disp_left.type() == CV_16UC1 && conf_left.type() == CV_16UC1))
	{
		std::cout << "Disp left and Conf left should be CV_16UC1" << std::endl;
		return -1;
	}
	if (!(plane_left.type() == CV_16UC3))
	{
		std::cout << "Plane left should be CV_16UC3" << std::endl;
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

	// 定义PMS匹配类实例
	start = std::chrono::steady_clock::now();
	PMSPropagation pmsp(width, height, pms_option, img_left, img_right, disp_left, conf_left, plane_left, edge_left);
	end = std::chrono::steady_clock::now();
	tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("Initialization: %.3f s\n", tt.count() / 1000000.0);

	start = std::chrono::steady_clock::now();
	for (int i = 0; i < num_iters; ++i)
	{
		pmsp.DoPropagation();
	}
	printf("Spatial: %.3f s; Refine: %.3f s\n", pmsp.time_spatial / 1000000.0, pmsp.time_refine / 1000000.0);
	end = std::chrono::steady_clock::now();
	tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("Propagation: %.3f s\n", tt.count() / 1000000.0);
	start = std::chrono::steady_clock::now();
	auto disp_prop_left = new float32[uint32(width * height)]();
	pmsp.GetDisparityFromPlane(disp_prop_left);
	end = std::chrono::steady_clock::now();
	tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("Get disparity: %.3f s\n", tt.count() / 1000000.0);

	// start = std::chrono::steady_clock::now();
	// uint16* disp_prop_left_u16 = new uint16[uint32(width * height)]();
	// for (int i=0;i<width*height;++i){
	// 	disp_prop_left_u16[i] = (uint16)(disp_prop_left[i]*500);
	// }
	// end = std::chrono::steady_clock::now();
	// tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	// printf("To uint16: %.3f s\n", tt.count() / 1000000.0);
	// start = std::chrono::steady_clock::now();
	// cv::Mat disp_prop_img(height,width, CV_16U, disp_prop_left_u16);
	// cv::imwrite(output_path, disp_prop_img);
	// end = std::chrono::steady_clock::now();
	// tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	// printf("Imwrite: %.3f s\n", tt.count() / 1000000.0);
	

	std::ofstream fout(output_path.c_str());

	if (!fout)
	{
		cout << "File Not Opened" << endl;
		return -1;
	}

	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			fout << disp_prop_left[i * width + j] << "\t";
		}
		fout << endl;
	}

	fout.close();
	return 0;
}