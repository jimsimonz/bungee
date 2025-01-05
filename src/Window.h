// Copyright (C) 2020-2025 Parabola Research Limited
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "Assert.h"
#include "Dispatch.h"
#include "Fourier.h"

#include <Eigen/Core>

#include <initializer_list>

namespace Bungee::Window {

Eigen::ArrayXf fromFrequencyDomainCoefficients(Fourier::Transforms &transforms, int log2Size, float gain, std::initializer_list<float> coefficients);

struct Apply
{
	template <bool add>
	static void special(const Eigen::Ref<const Eigen::ArrayXf> &window, const Eigen::Ref<const Eigen::ArrayXXf> &input, Eigen::Ref<Eigen::ArrayXXf> output);
};

typedef Dispatch<Apply, 2> DispatchApply;

} // namespace Bungee::Window
