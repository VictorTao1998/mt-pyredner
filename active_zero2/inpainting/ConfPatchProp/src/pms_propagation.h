/* -*-c++-*- PatchMatchStereo - Copyright (C) 2020.
 * Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
 *			  https://github.com/ethan-li-coding
 * Describe	: header of pms_propagation
 */

#ifndef PATCH_MATCH_STEREO_PROPAGATION_H_
#define PATCH_MATCH_STEREO_PROPAGATION_H_
#include "pms_types.h"

#include <random>
#include <chrono>
// opencv library
#include <opencv2/opencv.hpp>

/**
 * \brief 传播类
 */
class PMSPropagation
{
public:
	PMSPropagation(const sint32 &width, const sint32 &height, const PMSOption &option,
				   const cv::Mat &img_left, const cv::Mat &img_right,
				   const cv::Mat &disp_left, const cv::Mat &conf_left,
				   const cv::Mat &plane_left, const cv::Mat &edge_left);

	~PMSPropagation();

public:
	/** \brief 执行传播一次 */
	void DoPropagation();
	void GetDisparityFromPlane(float32 *disp);

private:
	/**
	 * \brief 空间传播
	 * \param x 像素x坐标
	 * \param y 像素y坐标
	 * \param direction 传播方向
	 */
	void SpatialPropagation(const sint32 &x, const sint32 &y, const sint32 &direction) const;

private:
	/** \brief PMS算法参数*/
	PMSOption option_;

	/** \brief 影像宽高 */
	sint32 width_;
	sint32 height_;

	/** \brief 传播迭代次数 */
	sint32 num_iter_;

	/** \brief 影像数据 */
	uint8 *img_left_;
	uint8 *img_right_;

	/** \brief 梯度数据 */
	PGradient *grad_left_;
	PGradient *grad_right_;

	/** \brief 平面数据 */
	DisparityPlane *plane_left_;
	bool *plane_conf_left_;

	bool *edge_left_;

	/** \brief 代价数据	 */
	float32 *cost_left_;

	/** \brief 视差数据 */
	float32 *disparity_map_;
	float32 *disparity_conf_;

	/** \brief 随机数生成器 */
	std::uniform_real_distribution<float32> *rand_disp_;
	std::uniform_real_distribution<float32> *rand_norm_;

public:
	uint64 time_spatial;
	uint64 time_refine;
};

#endif
