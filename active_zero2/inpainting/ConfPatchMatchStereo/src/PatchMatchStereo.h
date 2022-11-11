/* -*-c++-*- PatchMatchStereo - Copyright (C) 2020.
 * Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
 *			  https://github.com/ethan-li-coding
 * Describe	: header of patch match stereo class
 */

#pragma once
#include <vector>
#include <chrono>
#include "pms_types.h"
// opencv library
#include <opencv2/opencv.hpp>

/**
 * \brief PatchMatch类
 */
class PatchMatchStereo
{
public:
	PatchMatchStereo();
	~PatchMatchStereo();

public:
	/**
	 * \brief 类的初始化，完成一些内存的预分配、参数的预设置等
	 * \param width		输入，核线像对影像宽
	 * \param height		输入，核线像对影像高
	 * \param option		输入，算法参数
	 */
	bool Initialize(const sint32 &width, const sint32 &height, const PMSOption &option,
	 const cv::Mat &img_left, const cv::Mat &img_right, const cv::Mat &disp_left, const cv::Mat &disp_right,
	  const cv::Mat &plane_left, const cv::Mat &plane_right, const cv::Mat &edge_left, const cv::Mat &edge_right);

	bool Propagate(float32 *disp_prop_left);


	/**
	 * \brief 获取视差图指针
	 * \param view 0-左视图 1-右视图
	 * \return 视差图指针
	 */
	float *GetDisparityMap(const sint32 &view) const;

	/**
	 * \brief 获取梯度图指针
	 * \param view 0-左视图 1-右视图
	 * \return 梯度图指针
	 */
	PGradient *GetGradientMap(const sint32 &view) const;

private:
	
	/** \brief 随机初始化 */
	void RandomInitialization() const;

	/** \brief 计算梯度数据 */
	void ComputeGradient() const;

	/** \brief 迭代传播 */
	void Propagation() const;

	/** \brief 一致性检查	 */
	void LRCheck();

	/** \brief 视差图填充 */
	void FillHolesInDispMap();

	/** \brief 平面转换成视差 */
	void PlaneToDisparity() const;

	/** \brief 内存释放	 */
	void Release();

public:
	/** \brief 左影像视差图	*/
	float32 *disp_left_;
	/** \brief 右影像视差图	*/
	float32 *disp_right_;

private:
	/** \brief PMS参数	 */
	PMSOption option_;

	/** \brief 影像宽	 */
	sint32 width_;

	/** \brief 影像高	 */
	sint32 height_;

	/** \brief 左影像灰度数据	 */
	uint8 *img_left_;
	/** \brief 右影像灰度数据	 */
	uint8 *img_right_;

	/** disparity confidence mask */
	bool *disp_conf_left_;
	bool *disp_conf_right_;

	/** \brief 左影像平面集	*/
	DisparityPlane *plane_left_;
	/** \brief 右影像平面集	*/
	DisparityPlane *plane_right_;
	/** disparity plane confidence mask */
	bool *plane_conf_left_;
	bool *plane_conf_right_;

	/** edge mask */
	bool *edge_left_;
	bool *edge_right_;

	/** \brief 左影像梯度数据	 */
	PGradient *grad_left_;
	/** \brief 右影像梯度数据	 */
	PGradient *grad_right_;

	/** \brief 左影像聚合代价数据	 */
	float32 *cost_left_;
	/** \brief 右影像聚合代价数据	 */
	float32 *cost_right_;

	/** \brief 是否初始化标志	*/
	bool is_initialized_;

	/** \brief 误匹配区像素集	*/
	vector<pair<int, int>> mismatches_left_;
	vector<pair<int, int>> mismatches_right_;
};
