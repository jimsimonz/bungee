// Copyright (C) 2020-2025 Parabola Research Limited
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "Assert.h"
#include "Fourier.h"

#include <Eigen/Core>

namespace Bungee {

struct Input
{
	Eigen::ArrayXf analysisWindowBasic;
	Eigen::ArrayXXf windowedInput;

	Input(int log2SynthesisHop, int channelCount, Fourier::Transforms &transforms);

	int applyAnalysisWindow(const Eigen::Ref<const Eigen::ArrayXXf> &input);
};

} // namespace Bungee
