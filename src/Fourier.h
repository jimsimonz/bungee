// Copyright (C) 2020-2025 Parabola Research Limited
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "Assert.h"

#include <Eigen/Core>

#include <algorithm>
#include <complex>
#include <limits>
#include <memory>
#include <vector>

namespace Bungee {
extern const char *versionDescription;
}

namespace Bungee::Fourier {

inline constexpr int transformLength(int log2TransformLength)
{
	BUNGEE_ASSERT1(log2TransformLength >= 0);
	int length = 1 << log2TransformLength;
	BUNGEE_ASSERT1(length > 0);
	return length;
}

inline constexpr int binCount(int log2TransformLength)
{
	return transformLength(log2TransformLength - 1) + 1;
}

template <typename Scalar>
constexpr Scalar uninitialisedValue()
{
	return Scalar{};
}

template <>
constexpr float uninitialisedValue<float>()
{
	return std::numeric_limits<float>::signaling_NaN();
}

template <>
constexpr std::complex<float> uninitialisedValue<std::complex<float>>()
{
	return {uninitialisedValue<float>(), uninitialisedValue<float>()};
}

template <bool frequencyDomain, class T>
inline void resize(int log2TransformLength, int channelCount, T &array, int extra = 0)
{
	typedef typename T::Scalar Scalar;
	if constexpr (frequencyDomain)
	{
		auto pad = std::max<int>(1, EIGEN_DEFAULT_ALIGN_BYTES / std::min<int>(4, sizeof(Scalar)));
		array.resize(binCount(log2TransformLength) - 1 + pad + extra, channelCount);
	}
	else
	{
		array.resize(transformLength(log2TransformLength) + extra, channelCount);
	}

	if constexpr (Assert::level)
		array.setConstant(uninitialisedValue<Scalar>());
}

struct Transforms
{
	virtual ~Transforms() {}
	virtual void prepareForward(int log2TransformLength) = 0;
	virtual void prepareInverse(int log2TransformLength) = 0;
	virtual void forward(int log2TransformLength, const Eigen::Ref<const Eigen::ArrayXXf> &t, Eigen::Ref<Eigen::ArrayXXcf> f) const = 0;
	virtual void inverse(int log2TransformLength, Eigen::Ref<Eigen::ArrayXXf> t, const Eigen::Ref<const Eigen::ArrayXXcf> &f) const = 0;
};

// General case when an FFT implementation has different states for forward and reverse transforms of same size.
template <class F, class I>
struct KernelPair
{
	std::shared_ptr<F> f;
	std::shared_ptr<I> i;
	inline F *forward() const
	{
		return f.get();
	}
	inline I *inverse() const
	{
		return i.get();
	}
	void forward(F *x)
	{
		f.reset(x);
	}
	void inverse(I *x)
	{
		i.reset(x);
	}
};

// Special case when an FFT implementation can use the same state for forward and reverse transforms of same size.
template <class T>
struct KernelPair<T, T>
{
	std::shared_ptr<T> t;
	inline T *forward() const
	{
		return t.get();
	}
	inline T *inverse() const
	{
		return t.get();
	}
	void forward(T *x)
	{
		t.reset(x);
	}
	void inverse(T *x)
	{
		t.reset(x);
	}
};

template <class K, int log2MaxSize>
struct Cache :
	Transforms
{
	typedef KernelPair<typename K::Forward, typename K::Inverse> Entry;
	typedef std::array<Entry, log2MaxSize + 1> Table;

	Table table;

	void prepareForward(int log2TransformLength) override
	{
		if (!table[log2TransformLength].forward())
			table[log2TransformLength].forward(new typename K::Forward(log2TransformLength));
	}

	void prepareInverse(int log2TransformLength) override
	{
		if (!table[log2TransformLength].inverse())
			table[log2TransformLength].inverse(new typename K::Inverse(log2TransformLength));
	}

	void forward(int log2TransformLength, const Eigen::Ref<const Eigen::ArrayXXf> &t, Eigen::Ref<Eigen::ArrayXXcf> f) const override
	{
		BUNGEE_ASSERT1(t.cols() == t.cols());
		BUNGEE_ASSERT1(t.cols() == 1 || !t.IsRowMajor);
		BUNGEE_ASSERT1(f.cols() == 1 || !f.IsRowMajor);

		const auto transformLength = 1 << log2TransformLength;
		const auto &kernel = *table[log2TransformLength].forward();
		for (int c = 0; c < f.cols(); ++c)
			kernel.forward(log2TransformLength, (float *)t.col(c).topRows(transformLength).data(), f.col(c).topRows(transformLength / 2 + 1).data());
	}

	void inverse(int log2TransformLength, Eigen::Ref<Eigen::ArrayXXf> t, const Eigen::Ref<const Eigen::ArrayXXcf> &f) const override
	{
		BUNGEE_ASSERT1(t.cols() == t.cols());
		BUNGEE_ASSERT1(t.cols() == 1 || !t.IsRowMajor);
		BUNGEE_ASSERT1(f.cols() == 1 || !f.IsRowMajor);

		const auto transformLength = 1 << log2TransformLength;
		const auto &kernel = *table[log2TransformLength].inverse();
		for (int c = 0; c < f.cols(); ++c)
			kernel.inverse(log2TransformLength, t.col(c).topRows(transformLength).data(), (std::complex<float> *)f.col(c).topRows(transformLength / 2 + 1).data());
	}
};

std::unique_ptr<Transforms> transforms();

} // namespace Bungee::Fourier
