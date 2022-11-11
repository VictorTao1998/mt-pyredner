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
 * \brief PatchMatch��
 */
class PatchMatchStereo
{
public:
	PatchMatchStereo();
	~PatchMatchStereo();

public:
	/**
	 * \brief ��ĳ�ʼ�������һЩ�ڴ��Ԥ���䡢������Ԥ���õ�
	 * \param width		���룬�������Ӱ���
	 * \param height		���룬�������Ӱ���
	 * \param option		���룬�㷨����
	 */
	bool Initialize(const sint32 &width, const sint32 &height, const PMSOption &option,
	 const cv::Mat &img_left, const cv::Mat &img_right, const cv::Mat &disp_left, const cv::Mat &disp_right,
	  const cv::Mat &plane_left, const cv::Mat &plane_right, const cv::Mat &edge_left, const cv::Mat &edge_right);

	bool Propagate(float32 *disp_prop_left);


	/**
	 * \brief ��ȡ�Ӳ�ͼָ��
	 * \param view 0-����ͼ 1-����ͼ
	 * \return �Ӳ�ͼָ��
	 */
	float *GetDisparityMap(const sint32 &view) const;

	/**
	 * \brief ��ȡ�ݶ�ͼָ��
	 * \param view 0-����ͼ 1-����ͼ
	 * \return �ݶ�ͼָ��
	 */
	PGradient *GetGradientMap(const sint32 &view) const;

private:
	
	/** \brief �����ʼ�� */
	void RandomInitialization() const;

	/** \brief �����ݶ����� */
	void ComputeGradient() const;

	/** \brief �������� */
	void Propagation() const;

	/** \brief һ���Լ��	 */
	void LRCheck();

	/** \brief �Ӳ�ͼ��� */
	void FillHolesInDispMap();

	/** \brief ƽ��ת�����Ӳ� */
	void PlaneToDisparity() const;

	/** \brief �ڴ��ͷ�	 */
	void Release();

public:
	/** \brief ��Ӱ���Ӳ�ͼ	*/
	float32 *disp_left_;
	/** \brief ��Ӱ���Ӳ�ͼ	*/
	float32 *disp_right_;

private:
	/** \brief PMS����	 */
	PMSOption option_;

	/** \brief Ӱ���	 */
	sint32 width_;

	/** \brief Ӱ���	 */
	sint32 height_;

	/** \brief ��Ӱ��Ҷ�����	 */
	uint8 *img_left_;
	/** \brief ��Ӱ��Ҷ�����	 */
	uint8 *img_right_;

	/** disparity confidence mask */
	bool *disp_conf_left_;
	bool *disp_conf_right_;

	/** \brief ��Ӱ��ƽ�漯	*/
	DisparityPlane *plane_left_;
	/** \brief ��Ӱ��ƽ�漯	*/
	DisparityPlane *plane_right_;
	/** disparity plane confidence mask */
	bool *plane_conf_left_;
	bool *plane_conf_right_;

	/** edge mask */
	bool *edge_left_;
	bool *edge_right_;

	/** \brief ��Ӱ���ݶ�����	 */
	PGradient *grad_left_;
	/** \brief ��Ӱ���ݶ�����	 */
	PGradient *grad_right_;

	/** \brief ��Ӱ��ۺϴ�������	 */
	float32 *cost_left_;
	/** \brief ��Ӱ��ۺϴ�������	 */
	float32 *cost_right_;

	/** \brief �Ƿ��ʼ����־	*/
	bool is_initialized_;

	/** \brief ��ƥ�������ؼ�	*/
	vector<pair<int, int>> mismatches_left_;
	vector<pair<int, int>> mismatches_right_;
};
