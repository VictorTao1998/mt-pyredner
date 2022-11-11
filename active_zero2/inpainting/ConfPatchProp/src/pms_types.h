/* -*-c++-*- PatchMatchStereo - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
*			  https://github.com/ethan-li-coding
* Describe	: header of pms_types
*/

#ifndef PATCH_MATCH_STEREO_TYPES_H_
#define PATCH_MATCH_STEREO_TYPES_H_

#include <cstdint>
#include <limits>
#include <vector>
#include <math.h>
using std::vector;
using std::pair;

#ifndef SAFE_DELETE
#define SAFE_DELETE(P) {if(P) delete[](P);(P)=nullptr;}
#endif

/** \brief �������ͱ��� */
typedef int8_t			sint8;		// �з���8λ����
typedef uint8_t			uint8;		// �޷���8λ����
typedef int16_t			sint16;		// �з���16λ����
typedef uint16_t		uint16;		// �޷���16λ����
typedef int32_t			sint32;		// �з���32λ����
typedef uint32_t		uint32;		// �޷���32λ����
typedef int64_t			sint64;		// �з���64λ����
typedef uint64_t		uint64;		// �޷���64λ����
typedef float			float32;	// �����ȸ���
typedef double			float64;	// ˫���ȸ���

/** \brief float32��Чֵ */
constexpr auto Invalid_Float = std::numeric_limits<float32>::infinity();

/** \brief PMS�����ṹ�� */
struct PMSOption {
	sint32	patch_size;			// patch�ߴ磬�ֲ�����Ϊ patch_size*patch_size
	sint32  min_disparity;		// ��С�Ӳ�
	sint32	max_disparity;		// ����Ӳ�?
	float min_p;
	float max_p;

	float32	gamma;				// gamma Ȩֵ����
	float32	alpha;				// alpha ���ƶ�ƽ������
	float32	tau_col;			// tau for color	���ƶȼ�����ɫ�ռ�ľ��Բ���½ض���ֵ
	float32	tau_grad;			// tau for gradient ���ƶȼ����ݶȿռ�ľ��Բ��½ض����?

	
	PMSOption() : patch_size(35), min_disparity(0), max_disparity(64), min_p(-0.5), max_p(0.5), gamma(10.0f), alpha(0.9f), tau_col(10.0f),
	              tau_grad(2.0f){ }
};

/**
 * \brief ��ɫ�ṹ��
 */
struct PColor {
	uint8 r, g, b;
	PColor() : r(0), g(0), b(0) {}
	PColor(uint8 _b, uint8 _g, uint8 _r) {
		r = _r; g = _g; b = _b;
	}
};
/**
 * \brief �ݶȽṹ��
 */
struct PGradient {
	sint16 x, y;
	PGradient() : x(0), y(0) {}
	PGradient(sint16 _x, sint16 _y) {
		x = _x; y = _y;
	}
};

/**
* \brief ��άʸ���ṹ��
*/
struct PVector2f {

	float32 x = 0.0f, y = 0.0f;

	PVector2f() = default;
	PVector2f(const float32& _x, const float32& _y) {
		x = _x; y = _y;
	}
	PVector2f(const sint16& _x, const sint16& _y) {
		x = float32(_x); y = float32(_y);
	}
	PVector2f(const PVector2f& v) {
		x = v.x; y = v.y;
	}

	// ������operators
	// operator +
	PVector2f operator+(const PVector2f& v) const {
		return PVector2f(x + v.x, y + v.y);
	}
	// operator -
	PVector2f operator-(const PVector2f& v) const {
		return PVector2f(x - v.x, y - v.y);
	}
	// operator -t
	PVector2f operator-() const {
		return PVector2f(-x, -y);
	}
	// operator =
	PVector2f& operator=(const PVector2f& v) {
		if (this == &v) {
			return *this;
		}
		else {
			x = v.x; y = v.y;
			return *this;
		}
	}
};

/**
* \brief ��άʸ���ṹ��
*/
struct PVector3f {

	float32 x = 0.0f, y = 0.0f, z = 0.0f;

	PVector3f() = default;
	PVector3f(const float32& _x, const float32& _y, const float32& _z) {
		x = _x; y = _y; z = _z;
	}
	PVector3f(const uint8& _x, const uint8& _y, const uint8& _z) {
		x = float32(_x); y = float32(_y); z = float32(_z);
	}
	PVector3f(const PVector3f& v) {
		x = v.x; y = v.y; z = v.z;
	}

	// normalize
	void normalize() {
		if (x == 0.0f && y == 0.0f && z == 0.0f) {
			return;
		}
		else {
			const float32 sq = x * x + y * y + z * z;
			const float32 sqf = sqrt(sq);
			x /= sqf; y /= sqf; z /= sqf;
		}
	}

	// ������operators
	// operator +
	PVector3f operator+(const PVector3f& v) const {
		return PVector3f(x + v.x, y + v.y, z + v.z);
	}
	// operator -
	PVector3f operator-(const PVector3f& v) const {
		return PVector3f(x - v.x, y - v.y, z - v.z);
	}
	// operator -t
	PVector3f operator-() const {
		return PVector3f(-x, -y, -z);
	}
	// operator =
	PVector3f& operator=(const PVector3f& v) {
		if (this == &v) {
			return *this;
		}
		else {
			x = v.x;
			y = v.y;
			z = v.z;
			return *this;
		}
	}
	// operator ==
	bool operator==(const PVector3f& v) const {
		return (x == v.x) && (y == v.y) && (z == v.z);
	}
	// operator !=
	bool operator!=(const PVector3f& v) const {
		return (x != v.x) || (y != v.y) || (z != v.z);
	}

	// dot
	float32 dot(const PVector3f& v) const {
		return x * v.x + y * v.y + z * v.z;
	}
};

typedef  PVector3f PPoint3f;


/**
 * \brief �Ӳ�ƽ��
 */
struct DisparityPlane {
	PVector3f p;
	DisparityPlane() = default;
	DisparityPlane(const float32& x,const float32& y,const float32& z) {
		p.x = x; p.y = y; p.z = z;
	}
	DisparityPlane(const sint32& x, const sint32& y, const PVector3f& n, const float32& d) {
		p.x = -n.x / n.z;
		p.y = -n.y / n.z;
		p.z = (n.x * x + n.y * y + n.z * d) / n.z;
	}

	/**
	 * \brief ��ȡ��ƽ��������(x,y)���Ӳ�
	 * \param x		����x����
	 * \param y		����y����
	 * \return ����(x,y)���Ӳ�
	 */
	float32 to_disparity(const sint32& x,const sint32& y) const
	{
		return p.dot(PVector3f(float32(x), float32(y), 1.0f));
	}

	/** \brief ��ȡƽ��ķ���? */
	PVector3f to_normal() const
	{
		PVector3f n(p.x, p.y, -1.0f);
		n.normalize();
		return n;
	}

	/**
	 * \brief ���Ӳ�ƽ��ת������һ��ͼ
	 * ��������ͼƽ�淽��Ϊ d = a_p*xl + b_p*yl + c_p
	 * ������ͼ���㣺(1) xr = xl - d_p; (2) yr = yl; (3) �Ӳ������?(���������Ӳ�Ϊ��ֵ�����Ӳ�Ϊ��ֵ)
	 * ��������ͼ�Ӳ�ƽ�淽�̾Ϳɵõ�����ͼ����ϵ�µ�ƽ�淽��: d = -a_p*xr - b_p*yr - (c_p+a_p*d_p)
	 * ������ͬ��
	 * \param x		����x����
	 * \param y 	����y����
	 * \return ת�����ƽ��?
	 */
	DisparityPlane to_another_view(const sint32& x, const sint32& y) const
	{
		const float32 d = to_disparity(x, y);
		return { -p.x, -p.y, -p.z - p.x * d };
	}

	// operator ==
	bool operator==(const DisparityPlane& v) const {
		return p == v.p;
	}
	// operator !=
	bool operator!=(const DisparityPlane& v) const {
		return p != v.p;
	}
};

#endif
