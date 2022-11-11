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
 * \brief ������
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
	/** \brief ִ�д���һ�� */
	void DoPropagation();
	void GetDisparityFromPlane(float32 *disp);

private:
	/**
	 * \brief �ռ䴫��
	 * \param x ����x����
	 * \param y ����y����
	 * \param direction ��������
	 */
	void SpatialPropagation(const sint32 &x, const sint32 &y, const sint32 &direction) const;

private:
	/** \brief PMS�㷨����*/
	PMSOption option_;

	/** \brief Ӱ���� */
	sint32 width_;
	sint32 height_;

	/** \brief ������������ */
	sint32 num_iter_;

	/** \brief Ӱ������ */
	uint8 *img_left_;
	uint8 *img_right_;

	/** \brief �ݶ����� */
	PGradient *grad_left_;
	PGradient *grad_right_;

	/** \brief ƽ������ */
	DisparityPlane *plane_left_;
	bool *plane_conf_left_;

	bool *edge_left_;

	/** \brief ��������	 */
	float32 *cost_left_;

	/** \brief �Ӳ����� */
	float32 *disparity_map_;
	float32 *disparity_conf_;

	/** \brief ����������� */
	std::uniform_real_distribution<float32> *rand_disp_;
	std::uniform_real_distribution<float32> *rand_norm_;

public:
	uint64 time_spatial;
	uint64 time_refine;
};

#endif
