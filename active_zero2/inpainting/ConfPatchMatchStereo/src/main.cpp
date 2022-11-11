/*
reference: https://github.com/ethan-li-coding/PatchMatchStereo
*/

#include "stdafx.h"
#include <iostream>
#include "PatchMatchStereo.h"
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
	std::string disp_path_right = args[3];
	std::string plane_path_left = args[4];
	std::string plane_path_right = args[5];
	std::string edge_path_left = args[6];
	std::string edge_path_right = args[7];

	int patch_size = std::stoi(args[8]);
	int num_iters = std::stoi(args[9]);
	std::string output_path = args[10];
	auto start = std::chrono::steady_clock::now();
	cv::Mat img_left = cv::imread(img_path_left, 0);
	cv::Mat img_right = cv::imread(img_path_right, 0);

	if (img_left.data == nullptr || img_right.data == nullptr)
	{
		std::cout << "Fail to read images��" << std::endl;
		return -1;
	}
	cv::Mat disp_left = cv::imread(disp_path_left, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
	cv::Mat disp_right = cv::imread(disp_path_right, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

	cv::Mat plane_left = cv::imread(plane_path_left, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
	cv::Mat plane_right = cv::imread(plane_path_right, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

	cv::Mat edge_left = cv::imread(edge_path_left, 0);
	cv::Mat edge_right = cv::imread(edge_path_right, 0);
	auto end = std::chrono::steady_clock::now();
	auto tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	printf("Image reading: %.3f s\n", tt.count() / 1000000.0 );

	// size check
	if (!(disp_left.cols == img_left.cols && disp_left.rows == img_left.rows))
	{
		std::cout << "Disp left and Img Left should have same shape" << std::endl;
		return -1;
	}

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

	// PMSƥ��������
	PMSOption pms_option;
	// patch��С
	pms_option.patch_size = patch_size;
	// ��ѡ�ӲΧ
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
	// ������������
	pms_option.num_iters = num_iters;

	// һ���Լ��
	pms_option.is_check_lr = false;
	pms_option.lrcheck_thres = 1.0f;
	// �Ӳ�ͼ���
	pms_option.is_fill_holes = false;

	// ǰ��ƽ�д���
	pms_option.is_fource_fpw = false;

	// �����Ӳ��
	pms_option.is_integer_disp = false;

	// ����PMSƥ����ʵ��
	start = std::chrono::steady_clock::now();
	PatchMatchStereo pms;
	pms.Initialize(width, height, pms_option,
				   img_left, img_right, disp_left, disp_right,
				   plane_left, plane_right, edge_left, edge_right);
	end = std::chrono::steady_clock::now();
	tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("Initialization: %.3f s\n", tt.count() / 1000000.0 );

	int path_size = output_path.size();
	std::string init_disp_path = output_path.substr(0, path_size-4);
	init_disp_path += "_init.txt";
	std::ofstream finit(init_disp_path.c_str());

	if (!finit)
	{
		cout << "File Not Opened" << endl;
		return -1;
	}

	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			finit << pms.disp_left_[i * width + j] << "\t";
		}
		finit << endl;
	}

	finit.close();
	
	auto disp_prop_left = new float32[uint32(width * height)]();
	pms.Propagate(disp_prop_left);

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