/* -*-c++-*- PatchMatchStereo - Copyright (C) 2020.
 * Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
 *			  https://github.com/ethan-li-coding
 * Describe	: implement of pms_propagation
 */

#include "stdafx.h"
#include "pms_propagation.h"

PMSPropagation::PMSPropagation(const sint32 &width, const sint32 &height, const PMSOption &option,
							   const cv::Mat &img_left, const cv::Mat &img_right,
							   const cv::Mat &disp_left, const cv::Mat &conf_left,
							   const cv::Mat &plane_left, const cv::Mat &edge_left)
{
	width_ = width;
	height_ = height;
	// PMS参数
	option_ = option;

	//··· 开辟内存空间
	const sint32 img_size = width * height;

	// 灰度数据
	img_left_ = new uint8[img_size];
	img_right_ = new uint8[img_size];

	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			img_left_[i * width + j] = img_left.at<uint8>(i, j);
			img_right_[i * width + j] = img_right.at<uint8>(i, j);
		}
	}

	// 视差图
	disparity_map_ = new float32[img_size];
	disparity_conf_ = new float32[img_size];

	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			int p = i * width + j;
			float32 disp = (float32)disp_left.at<uint16>(i, j) / 500.0;
			float32 conf = (float32)conf_left.at<uint16>(i, j) / 65535.0;
			disparity_map_[p] = disp;
			disparity_conf_[p] = conf;
		}
	}

	plane_left_ = new DisparityPlane[img_size];
	plane_conf_left_ = new bool[img_size];
	int plane_conf_left_num = 0;
	// 代价数据
	cost_left_ = new float32[img_size];

	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if (plane_left.at<cv::Vec3w>(i, j)[2] == 1)
			{
				float32 px = ((float32)plane_left.at<cv::Vec3w>(i, j)[0] - 30000.0) / 30000.0;
				float32 py = ((float32)plane_left.at<cv::Vec3w>(i, j)[1] - 30000.0) / 30000.0;
				plane_left_[i * width + j] = DisparityPlane(px, py, disparity_map_[i * width + j] - px * j - py * i);
				plane_conf_left_[i * width + j] = true;
				plane_conf_left_num += 1;
				cost_left_[i * width + j] = 0.0;
			}
			else
			{
				plane_conf_left_[i * width + j] = false;
				plane_left_[i * width + j] = DisparityPlane(-1.0, -1.0, 0.0);
				cost_left_[i * width + j] = FLT_MAX;
			}
		}
	}
	std::cout << "plane_conf_left_num: " << plane_conf_left_num << std::endl;

	// edge mask
	int edge_left_num = 0;
	edge_left_ = new bool[img_size];
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			edge_left_[i * width + j] = (edge_left.at<uint8>(i, j) > 0);
			if (edge_left_[i * width + j])
			{
				edge_left_num += 1;
			}
		}
	}
	printf("Left edge num: %d\n", edge_left_num);

	// 图像梯度数据
	grad_left_ = new PGradient[img_size]();
	grad_right_ = new PGradient[img_size]();

	time_spatial = 0;
	time_refine = 0;
}

PMSPropagation::~PMSPropagation()
{
	if (rand_disp_)
	{
		delete rand_disp_;
		rand_disp_ = nullptr;
	}
	if (rand_norm_)
	{
		delete rand_norm_;
		rand_norm_ = nullptr;
	}
}

void PMSPropagation::DoPropagation()
{
	if (!img_left_ || !img_right_ || !grad_left_ || !grad_right_ || !cost_left_ || !plane_left_ || !disparity_map_ || !disparity_conf_ || !rand_disp_ || !rand_norm_)
	{
		return;
	}

	const sint32 dir = (num_iter_ % 2 == 0) ? 1 : -1;
	sint32 y = (dir == 1) ? 0 : height_ - 1;
	for (sint32 i = 0; i < height_; i++)
	{
		sint32 x = (dir == 1) ? 0 : width_ - 1;
		for (sint32 j = 0; j < width_; j++)
		{

			// 空间传播
			auto start = std::chrono::steady_clock::now();
			SpatialPropagation(x, y, dir);
			auto end = std::chrono::steady_clock::now();
			auto tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			time_spatial += tt.count();

			x += dir;
		}
		y += dir;
	}
	++num_iter_;
}

void PMSPropagation::SpatialPropagation(const sint32 &x, const sint32 &y, const sint32 &direction) const
{
	// ---
	// 空间传播
	if (plane_conf_left_[y * width_ + x])
	{
		return;
	}

	// 偶数次迭代从左上到右下传播
	// 奇数次迭代从右下到左上传播
	const sint32 dir = direction;

	// 获取p当前的视差平面并计算代价
	auto &plane_p = plane_left_[y * width_ + x];

	// 获取p左(右)侧像素的视差平面，计算将平面分配给p时的代价，取较小值
	const sint32 xd = x - dir;
	if (xd >= 0 && xd < width_)
	{
		if (!edge_left_[y * width_ + xd])
		{
			auto &plane = plane_left_[y * width_ + xd];
			if (plane.p.x > -0.9)
			{
				if (plane_p.p.x > -0.9)
				{
					plane_p.p.x = (plane_p.p.x + plane.p.x) / 2;
					plane_p.p.y = (plane_p.p.y + plane.p.y) / 2;
					plane_p.p.z = (plane_p.p.z + plane.p.z) / 2;
				}
				else
				{
					plane_p = plane;
				}
			}
		}
	}

	// 获取p上(下)侧像素的视差平面，计算将平面分配给p时的代价，取较小值
	const sint32 yd = y - dir;
	if (yd >= 0 && yd < height_)
	{
		if (!edge_left_[y * width_ + xd])
		{
			auto &plane = plane_left_[yd * width_ + x];
			if (plane.p.x > -0.9)
			{
				if (plane_p.p.x > -0.9)
				{
					plane_p.p.x = (plane_p.p.x + plane.p.x) / 2;
					plane_p.p.y = (plane_p.p.y + plane.p.y) / 2;
					plane_p.p.z = (plane_p.p.z + plane.p.z) / 2;
				}
				else
				{
					plane_p = plane;
				}
			}
		}
	}
}

void PMSPropagation::GetDisparityFromPlane(float32 *disp)
{
	const sint32 width = width_;
	const sint32 height = height_;

	for (sint32 y = 0; y < height; y++)
	{
		for (sint32 x = 0; x < width; x++)
		{
			const sint32 p = y * width + x;
			const auto &plane = plane_left_[p];
			disp[p] = plane.to_disparity(x, y);
		}
	}
}