/* -*-c++-*- PatchMatchStereo - Copyright (C) 2020.
 * Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
 *			  https://github.com/ethan-li-coding
 * Describe	: implement of patch match stereo class
 */

#include "stdafx.h"
#include "PatchMatchStereo.h"
#include <ctime>
#include <random>
#include <cmath>
#include "pms_propagation.h"
#include "pms_util.h"

PatchMatchStereo::PatchMatchStereo() : width_(0), height_(0), img_left_(nullptr), img_right_(nullptr),
									   disp_left_(nullptr), disp_right_(nullptr),
									   disp_conf_left_(nullptr), disp_conf_right_(nullptr),
									   plane_left_(nullptr), plane_right_(nullptr),
									   plane_conf_left_(nullptr), plane_conf_right_(nullptr),
									   edge_left_(nullptr), edge_right_(nullptr),
									   grad_left_(nullptr), grad_right_(nullptr),
									   cost_left_(nullptr), cost_right_(nullptr),
									   is_initialized_(false) {}

PatchMatchStereo::~PatchMatchStereo()
{
	Release();
}

bool PatchMatchStereo::Initialize(const sint32 &width, const sint32 &height, const PMSOption &option,
								  const cv::Mat &img_left, const cv::Mat &img_right, const cv::Mat &disp_left, const cv::Mat &disp_right,
								  const cv::Mat &plane_left, const cv::Mat &plane_right, const cv::Mat &edge_left, const cv::Mat &edge_right)
{
	// ··· 赋值

	// 影像尺寸
	width_ = width;
	height_ = height;
	// PMS参数
	option_ = option;

	if (width <= 0 || height <= 0)
	{
		return false;
	}

	//··· 开辟内存空间
	const sint32 img_size = width * height;
	const sint32 disp_range = option.max_disparity - option.min_disparity;

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

	// edge mask
	int edge_left_num = 0;
	int edge_right_num = 0;
	edge_left_ = new bool[img_size];
	edge_right_ = new bool[img_size];
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			edge_left_[i * width + j] = (edge_left.at<uint8>(i, j) > 0);
			edge_right_[i * width + j] = (edge_right.at<uint8>(i, j) > 0);
			if (edge_left_[i * width + j])
			{
				edge_left_num += 1;
			}
			if (edge_right_[i * width + j])
			{
				edge_right_num += 1;
			}
		}
	}
	printf("Left edge num: %d, Right edge num: %d\n", edge_left_num, edge_right_num);

	// 视差图
	disp_left_ = new float32[img_size];
	disp_right_ = new float32[img_size];
	// disparity confidence mask
	disp_conf_left_ = new bool[img_size];
	disp_conf_right_ = new bool[img_size];

	float32 disp_prev = 0.0;
	bool count_edge = false;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			float p = (float)disp_left.at<uint16>(i, j) / 500.0;
			if (p > 0.01)
			{
				count_edge = false;
				disp_prev = p;
				disp_conf_left_[i * width + j] = true;
			}
			else
			{
				disp_conf_left_[i * width + j] = false;
				count_edge = count_edge | edge_left_[i * width + j];
				if (count_edge)
				{
					disp_prev = 0.0;
				}
			}
			disp_left_[i * width + j] = disp_prev;
		}
	}
	disp_prev = 0.0;
	count_edge = false;
	for (int j = width - 1; j > -1; --j)
	{
		for (int i = height - 1; i > -1; --i)
		{
			if (disp_conf_left_[i * width + j])
			{
				disp_prev = disp_left_[i * width + j];
				count_edge = false;
			}
			else
			{
				count_edge = count_edge | edge_left_[i * width + j];
				if ((!count_edge) && (disp_left_[i * width + j] < 0.01))
				{
					disp_left_[i * width + j] = disp_prev;
				}
			}
		}
	}
	disp_prev = 0.0;
	count_edge = false;
	for (int i = 0; i < height; ++i)
	{
		for (int j = width - 1; j > -1; --j)
		{
			if (disp_conf_left_[i * width + j])
			{
				disp_prev = disp_left_[i * width + j];
				count_edge = false;
			}
			else
			{
				count_edge = count_edge | edge_left_[i * width + j];
				if ((!count_edge) && (disp_left_[i * width + j] < 0.01))
				{
					disp_left_[i * width + j] = disp_prev;
				}
			}
		}
	}
	disp_prev = 0.0;
	count_edge = false;
	for (int j = width - 1; j > -1; --j)
	{
		for (int i = 0; i < height; ++i)
		{
			if (disp_conf_left_[i * width + j])
			{
				disp_prev = disp_left_[i * width + j];
				count_edge = false;
			}
			else
			{
				count_edge = count_edge | edge_left_[i * width + j];
				if ((!count_edge) && (disp_left_[i * width + j] < 0.01))
				{
					disp_left_[i * width + j] = disp_prev;
				}
			}
		}
	}

	// right image
	disp_prev = 0.0;
	count_edge = false;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			float p = -(float)disp_right.at<uint16>(i, j) / 500.0;
			if (p < -0.01)
			{
				count_edge = false;
				disp_prev = p;
				disp_conf_right_[i * width + j] = true;
			}
			else
			{
				disp_conf_right_[i * width + j] = false;
				count_edge = count_edge | edge_right_[i * width + j];
				if (count_edge)
				{
					disp_prev = 0.0;
				}
			}
			disp_right_[i * width + j] = disp_prev;
		}
	}
	disp_prev = 0.0;
	count_edge = false;
	for (int j = width - 1; j > -1; --j)
	{
		for (int i = height - 1; i > -1; --i)
		{
			if (disp_conf_right_[i * width + j])
			{
				disp_prev = disp_right_[i * width + j];
				count_edge = false;
			}
			else
			{
				count_edge = count_edge | edge_right_[i * width + j];
				if ((!count_edge) && (disp_right_[i * width + j] > -0.01))
				{
					disp_right_[i * width + j] = disp_prev;
				}
			}
		}
	}
	disp_prev = 0.0;
	count_edge = false;
	for (int i = 0; i < height; ++i)
	{
		for (int j = width - 1; j > -1; --j)
		{
			if (disp_conf_right_[i * width + j])
			{
				disp_prev = disp_right_[i * width + j];
				count_edge = false;
			}
			else
			{
				count_edge = count_edge | edge_right_[i * width + j];
				if ((!count_edge) && (disp_right_[i * width + j] > -0.01))
				{
					disp_right_[i * width + j] = disp_prev;
				}
			}
		}
	}
	disp_prev = 0.0;
	count_edge = false;
	for (int j = width - 1; j > -1; --j)
	{
		for (int i = 0; i < height; ++i)
		{
			if (disp_conf_right_[i * width + j])
			{
				disp_prev = disp_right_[i * width + j];
				count_edge = false;
			}
			else
			{
				count_edge = count_edge | edge_right_[i * width + j];
				if ((!count_edge) && (disp_right_[i * width + j] > -0.01))
				{
					disp_right_[i * width + j] = disp_prev;
				}
			}
		}
	}
	// check
	float32 disp_left_min = 10000;
	float32 disp_left_max = -10000;
	float32 disp_right_min = 10000;
	float32 disp_right_max = -10000;
	for (int i = 0; i < img_size; ++i)
	{
		if (disp_left_[i] < disp_left_min)
		{
			disp_left_min = disp_left_[i];
		}
		if (disp_left_[i] > disp_left_max)
		{
			disp_left_max = disp_left_[i];
		}
		if (disp_right_[i] < disp_right_min)
		{
			disp_right_min = disp_right_[i];
		}
		if (disp_right_[i] > disp_right_max)
		{
			disp_right_max = disp_right_[i];
		}
	}

	printf("disp left: min: %.2f, max: %.2f\n", disp_left_min, disp_left_max);
	printf("disp right: min: %.2f, max: %.2f\n", disp_right_min, disp_right_max);

	// 平面集
	plane_left_ = new DisparityPlane[img_size];
	plane_right_ = new DisparityPlane[img_size];
	// plane confidence mask
	int plane_conf_left_num = 0;
	int plane_conf_right_num = 0;
	plane_conf_left_ = new bool[img_size];
	plane_conf_right_ = new bool[img_size];

	float32 prev_px = 0.0;
	float32 prev_py = 0.0;
	count_edge = false;

	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if (plane_left.at<cv::Vec3w>(i, j)[2] != 10000)
			{
				count_edge = false;
				PVector3f n;
				n.x = ((float32)plane_left.at<cv::Vec3w>(i, j)[0] - 10000.0) / 10000.0;
				n.y = ((float32)plane_left.at<cv::Vec3w>(i, j)[1] - 10000.0) / 10000.0;
				n.z = ((float32)plane_left.at<cv::Vec3w>(i, j)[2] - 10000.0) / 10000.0;
				plane_left_[i * width + j] = DisparityPlane((sint32)j, (sint32)i, n, disp_left_[i * width + j]);
				prev_px = plane_left_[i * width + j].p.x;
				prev_py = plane_left_[i * width + j].p.y;
				plane_conf_left_[i * width + j] = true;
				plane_conf_left_num += 1;
			}
			else
			{
				plane_conf_left_[i * width + j] = false;
				count_edge = count_edge | edge_left_[i * width + j];
				if (count_edge)
				{
					prev_px = 0.0;
					prev_py = 0.0;
				}
				plane_left_[i * width + j] = DisparityPlane(prev_px, prev_py, disp_left_[i * width + j]);
			}
		}
	}

	prev_px = 0.0;
	prev_py = 0.0;
	count_edge = false;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if (plane_right.at<cv::Vec3w>(i, j)[2] != 10000)
			{
				count_edge = false;
				PVector3f n;
				n.x = ((float32)plane_right.at<cv::Vec3w>(i, j)[0] - 10000.0) / 10000.0;
				n.y = ((float32)plane_right.at<cv::Vec3w>(i, j)[1] - 10000.0) / 10000.0;
				n.z = ((float32)plane_right.at<cv::Vec3w>(i, j)[2] - 10000.0) / 10000.0;
				plane_right_[i * width + j] = DisparityPlane((sint32)j, (sint32)i, n, disp_right_[i * width + j]);
				prev_px = plane_right_[i * width + j].p.x;
				prev_py = plane_right_[i * width + j].p.y;
				plane_conf_right_[i * width + j] = true;
				plane_conf_right_num += 1;
			}
			else
			{
				plane_conf_right_[i * width + j] = false;
				count_edge = count_edge | edge_right_[i * width + j];
				if (count_edge)
				{
					prev_px = 0.0;
					prev_py = 0.0;
				}
				plane_right_[i * width + j] = DisparityPlane(prev_px, prev_py, disp_right_[i * width + j]);
			}
		}
	}
	std::cout << "plane_conf_left_num: " << plane_conf_left_num << "; plane_conf_right_num: " << plane_conf_right_num << std::endl;

	// 梯度数据
	grad_left_ = new PGradient[img_size]();
	grad_right_ = new PGradient[img_size]();
	// 代价数据
	cost_left_ = new float32[img_size];
	cost_right_ = new float32[img_size];

	is_initialized_ = img_left_ && img_right_ && disp_left_ && disp_right_ && plane_left_ && plane_right_ && grad_left_ && grad_right_ && cost_left_ && cost_right_;

	return is_initialized_;
}

void PatchMatchStereo::Release()
{
	SAFE_DELETE(disp_left_);
	SAFE_DELETE(disp_right_);
	SAFE_DELETE(disp_conf_left_);
	SAFE_DELETE(disp_conf_right_);
	SAFE_DELETE(plane_left_);
	SAFE_DELETE(plane_right_);
	SAFE_DELETE(plane_conf_left_);
	SAFE_DELETE(plane_conf_right_);
	SAFE_DELETE(edge_left_);
	SAFE_DELETE(edge_right_);
	SAFE_DELETE(grad_left_);
	SAFE_DELETE(grad_right_);
	SAFE_DELETE(cost_left_);
	SAFE_DELETE(cost_right_);
}

bool PatchMatchStereo::Propagate(float32 *disp_prop_left)
{
	if (!is_initialized_)
	{
		return false;
	}

	auto start = std::chrono::steady_clock::now();
	// 随机初始化
	RandomInitialization();

	// 计算梯度图
	ComputeGradient();
	auto end = std::chrono::steady_clock::now();
	auto tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("Preprocess: %.3f s\n", tt.count() / 1000000.0);

	// 迭代传播
	Propagation();

	start = std::chrono::steady_clock::now();
	// 平面转换成视差
	PlaneToDisparity();

	// 左右一致性检查
	if (option_.is_check_lr)
	{
		// 一致性检查
		LRCheck();
	}

	// 视差填充
	if (option_.is_fill_holes)
	{
		FillHolesInDispMap();
	}
	end = std::chrono::steady_clock::now();
	tt = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	printf("Postprocess: %.3f s\n", tt.count() / 1000000.0);

	// 输出视差图
	if (disp_prop_left && disp_left_)
	{
		memcpy(disp_prop_left, disp_left_, height_ * width_ * sizeof(float32));
	}
	return true;
}

float *PatchMatchStereo::GetDisparityMap(const sint32 &view) const
{
	switch (view)
	{
	case 0:
		return disp_left_;
	case 1:
		return disp_right_;
	default:
		return nullptr;
	}
}

PGradient *PatchMatchStereo::GetGradientMap(const sint32 &view) const
{
	switch (view)
	{
	case 0:
		return grad_left_;
	case 1:
		return grad_right_;
	default:
		return nullptr;
	}
}

void PatchMatchStereo::RandomInitialization() const
{
	const sint32 width = width_;
	const sint32 height = height_;
	if (width <= 0 || height <= 0 ||
		disp_left_ == nullptr || disp_right_ == nullptr ||
		plane_left_ == nullptr || plane_right_ == nullptr)
	{
		return;
	}
	const auto &option = option_;
	const sint32 min_disparity = option.min_disparity;
	const sint32 max_disparity = option.max_disparity;

	// 随机数生成器
	std::random_device rd;
	std::mt19937 gen;
	gen.seed(rd());
	std::uniform_real_distribution<float32> rand_d(static_cast<float32>(min_disparity), static_cast<float32>(max_disparity));
	std::uniform_real_distribution<float32> rand_n(-1.0f, 1.0f);

	for (int k = 0; k < 2; k++)
	{
		auto *disp_ptr = k == 0 ? disp_left_ : disp_right_;
		auto *disp_conf_ptr = k == 0 ? disp_conf_left_ : disp_conf_right_;
		auto *plane_ptr = k == 0 ? plane_left_ : plane_right_;
		auto *plane_conf_ptr = k == 0 ? plane_conf_left_ : plane_conf_right_;
		sint32 sign = (k == 0) ? 1 : -1;
		auto &plane_prev = plane_ptr[0];
		for (sint32 y = 0; y < height; y++)
		{
			for (sint32 x = 0; x < width; x++)
			{
				const sint32 p = y * width + x;
				if (disp_conf_ptr[p])
				{
					if (option.is_integer_disp)
					{
						disp_ptr[p] = static_cast<float32>(round(disp_ptr[p]));
					}
				}
				else
				{
					// random disparity
					float32 disp = sign * rand_d(gen);
					if (option.is_integer_disp)
					{
						disp = static_cast<float32>(round(disp));
					}
					disp_ptr[p] = disp;
				}

				if (!plane_conf_ptr[p])
				{
					// random normal
					PVector3f norm;
					if (!option.is_fource_fpw)
					{
						norm.x = rand_n(gen);
						norm.y = rand_n(gen);
						float32 z = rand_n(gen);
						while (z == 0.0f)
						{
							z = rand_n(gen);
						}
						norm.z = z;
						norm.normalize();
					}
					else
					{
						norm.x = 0.0f;
						norm.y = 0.0f;
						norm.z = 1.0f;
					}

					// 计算视差平面
					plane_ptr[p] = DisparityPlane(x, y, norm, disp_ptr[p]);
				}
			}
		}
	}
}

void PatchMatchStereo::ComputeGradient() const
{
	const sint32 width = width_;
	const sint32 height = height_;
	if (width <= 0 || height <= 0 ||
		grad_left_ == nullptr || grad_right_ == nullptr ||
		img_left_ == nullptr || img_right_ == nullptr)
	{
		return;
	}

	// Sobel梯度算子
	for (sint32 n = 0; n < 2; n++)
	{
		auto *gray = (n == 0) ? img_left_ : img_right_;
		auto *grad = (n == 0) ? grad_left_ : grad_right_;
		for (int y = 1; y < height - 1; y++)
		{
			for (int x = 1; x < width - 1; x++)
			{
				const auto grad_x = (-gray[(y - 1) * width + x - 1] + gray[(y - 1) * width + x + 1]) +
									(-2 * gray[y * width + x - 1] + 2 * gray[y * width + x + 1]) +
									(-gray[(y + 1) * width + x - 1] + gray[(y + 1) * width + x + 1]);
				const auto grad_y = (-gray[(y - 1) * width + x - 1] - 2 * gray[(y - 1) * width + x] - gray[(y - 1) * width + x + 1]) +
									(gray[(y + 1) * width + x - 1] + 2 * gray[(y + 1) * width + x] + gray[(y + 1) * width + x + 1]);

				// 这里除以8是为了让梯度的最大值不超过255，这样计算代价时梯度差和颜色差位于同一个尺度
				grad[y * width + x].x = grad_x / 8;
				grad[y * width + x].y = grad_y / 8;
			}
		}
	}
}

void PatchMatchStereo::Propagation() const
{
	const sint32 width = width_;
	const sint32 height = height_;
	if (width <= 0 || height <= 0 ||
		img_left_ == nullptr || img_right_ == nullptr ||
		grad_left_ == nullptr || grad_right_ == nullptr ||
		disp_left_ == nullptr || disp_right_ == nullptr ||
		plane_left_ == nullptr || plane_right_ == nullptr)
	{
		return;
	}

	// 左右视图匹配参数
	const auto opion_left = option_;
	auto option_right = option_;
	option_right.min_disparity = -opion_left.max_disparity;
	option_right.max_disparity = -opion_left.min_disparity;

	// 左右视图传播实例
	PMSPropagation propa_left(width, height, img_left_, img_right_, grad_left_, grad_right_,
							  plane_left_, plane_right_, plane_conf_left_, plane_conf_right_,
							  opion_left, cost_left_, cost_right_, disp_left_, disp_conf_left_);
	PMSPropagation propa_right(width, height, img_right_, img_left_, grad_right_, grad_left_,
							   plane_right_, plane_left_, plane_conf_right_, plane_conf_left_,
							   option_right, cost_right_, cost_left_, disp_right_, disp_conf_right_);

	// 迭代传播
	for (int k = 0; k < option_.num_iters; k++)
	{
		printf("Iter %d/%d.\n", k + 1, option_.num_iters);
		propa_left.DoPropagation();
		printf("Left: spatial %.3f s, refine %.3f s, view %.3f s\n",
			   propa_left.time_spatial / 1000000.0, propa_left.time_refine / 1000000.0, propa_left.time_view / 1000000.0);
		propa_right.DoPropagation();
		printf("Right: spatial %.3f s, refine %.3f s, view %.3f s\n",
			   propa_right.time_spatial / 1000000.0, propa_right.time_refine / 1000000.0, propa_right.time_view / 1000000.0);
	}
}

void PatchMatchStereo::LRCheck()
{
	const sint32 width = width_;
	const sint32 height = height_;

	const float32 &threshold = option_.lrcheck_thres;

	// k==0 : 左视图一致性检查
	// k==1 : 右视图一致性检查
	for (int k = 0; k < 2; k++)
	{
		auto *disp_left = (k == 0) ? disp_left_ : disp_right_;
		auto *disp_right = (k == 0) ? disp_right_ : disp_left_;
		auto &mismatches = (k == 0) ? mismatches_left_ : mismatches_right_;
		mismatches.clear();

		// ---左右一致性检查
		for (sint32 y = 0; y < height; y++)
		{
			for (sint32 x = 0; x < width; x++)
			{

				// 左影像视差值
				auto &disp = disp_left[y * width + x];

				if (disp == Invalid_Float)
				{
					mismatches.emplace_back(x, y);
					continue;
				}

				// 根据视差值找到右影像上对应的同名像素
				const auto col_right = lround(x - disp);

				if (col_right >= 0 && col_right < width)
				{
					// 右影像上同名像素的视差值
					auto &disp_r = disp_right[y * width + col_right];

					// 判断两个视差值是否一致（差值在阈值内为一致）
					// 在本代码里，左右视图的视差值符号相反
					if (abs(disp + disp_r) > threshold)
					{
						// 让视差值无效
						disp = Invalid_Float;
						mismatches.emplace_back(x, y);
					}
				}
				else
				{
					// 通过视差值在右影像上找不到同名像素（超出影像范围）
					disp = Invalid_Float;
					mismatches.emplace_back(x, y);
				}
			}
		}
	}
}

void PatchMatchStereo::FillHolesInDispMap()
{
	const sint32 width = width_;
	const sint32 height = height_;
	if (width <= 0 || height <= 0 ||
		disp_left_ == nullptr || disp_right_ == nullptr ||
		plane_left_ == nullptr || plane_right_ == nullptr)
	{
		return;
	}

	const auto &option = option_;

	// k==0 : 左视图视差填充
	// k==1 : 右视图视差填充
	for (int k = 0; k < 2; k++)
	{
		auto &mismatches = (k == 0) ? mismatches_left_ : mismatches_right_;
		if (mismatches.empty())
		{
			continue;
		}
		const auto *img_ptr = (k == 0) ? img_left_ : img_right_;
		const auto *plane_ptr = (k == 0) ? plane_left_ : plane_right_;
		auto *disp_ptr = (k == 0) ? disp_left_ : disp_right_;
		vector<float32> fill_disps(mismatches.size()); // 存储每个待填充像素的视差
		for (auto n = 0u; n < mismatches.size(); n++)
		{
			auto &pix = mismatches[n];
			const sint32 x = pix.first;
			const sint32 y = pix.second;
			vector<DisparityPlane> planes;

			// 向左向右各搜寻第一个有效像素，记录平面
			sint32 xs = x + 1;
			while (xs < width)
			{
				if (disp_ptr[y * width + xs] != Invalid_Float)
				{
					planes.push_back(plane_ptr[y * width + xs]);
					break;
				}
				xs++;
			}
			xs = x - 1;
			while (xs >= 0)
			{
				if (disp_ptr[y * width + xs] != Invalid_Float)
				{
					planes.push_back(plane_ptr[y * width + xs]);
					break;
				}
				xs--;
			}

			if (planes.empty())
			{
				continue;
			}
			else if (planes.size() == 1u)
			{
				fill_disps[n] = planes[0].to_disparity(x, y);
			}
			else
			{
				// 选择较小的视差
				const auto d1 = planes[0].to_disparity(x, y);
				const auto d2 = planes[1].to_disparity(x, y);
				fill_disps[n] = abs(d1) < abs(d2) ? d1 : d2;
			}
		}
		for (auto n = 0u; n < mismatches.size(); n++)
		{
			auto &pix = mismatches[n];
			const sint32 x = pix.first;
			const sint32 y = pix.second;
			disp_ptr[y * width + x] = fill_disps[n];
		}

		// 加权中值滤波
		pms_util::WeightedMedianFilter(img_ptr, width, height, option.patch_size, option.gamma, mismatches, disp_ptr);
	}
}

void PatchMatchStereo::PlaneToDisparity() const
{
	const sint32 width = width_;
	const sint32 height = height_;
	if (width <= 0 || height <= 0 ||
		disp_left_ == nullptr || disp_right_ == nullptr ||
		plane_left_ == nullptr || plane_right_ == nullptr)
	{
		return;
	}
	for (int k = 0; k < 2; k++)
	{
		auto *plane_ptr = (k == 0) ? plane_left_ : plane_right_;
		auto *disp_ptr = (k == 0) ? disp_left_ : disp_right_;
		for (sint32 y = 0; y < height; y++)
		{
			for (sint32 x = 0; x < width; x++)
			{
				const sint32 p = y * width + x;
				const auto &plane = plane_ptr[p];
				disp_ptr[p] = plane.to_disparity(x, y);
			}
		}
	}
}
