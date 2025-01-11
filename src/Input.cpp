// Copyright (C) 2020-2025 Parabola Research Limited
// SPDX-License-Identifier: MPL-2.0

#include "Input.h"
#include "Grain.h"
#include "Instrumentation.h"
#include "log2.h"

#include <numbers>

namespace Bungee {

using namespace Internal;

namespace {
static constexpr float pi = std::numbers::pi_v<float>;
static constexpr float gain = (3 * pi) / (3 * pi + 8);
} // namespace

Input::Input(int log2SynthesisHop, int channelCount, Fourier::Transforms &transforms) :
	window(Window::fromFrequencyDomainCoefficients(transforms, log2SynthesisHop + 3, gain / (8 << log2SynthesisHop), {1.f, 0.5f})),
	windowedInput{(8 << log2SynthesisHop), channelCount}
{
	windowedInput.setZero();
	transforms.prepareForward(log2SynthesisHop + 3);
}

int Input::applyAnalysisWindow(const Eigen::Ref<const Eigen::ArrayXXf> &input, int muteFrameCountHead, int muteFrameCountTail)
{
	const int half = window.rows() / 2;
	BUNGEE_ASSERT1(input.rows() % 2 == 0);
	const int unused = std::max<int>(input.rows() / 2 - half, 0);
	muteFrameCountHead -= unused;
	muteFrameCountTail -= unused;

	{
		// top half of window, bottom half of input -> top half of output
		const int muteHead = std::clamp(muteFrameCountHead - half, 0, half);
		const int muteTail = std::clamp(muteFrameCountTail, 0, half);
		const int unmuted = half - muteHead - muteTail;

		windowedInput.topRows(muteHead).setZero();

		Window::Apply::special<false>(
			window.segment(muteHead, unmuted),
			input.middleRows(input.rows() / 2 + muteHead, unmuted),
			windowedInput.middleRows(muteHead, unmuted));

		windowedInput.middleRows(half - muteTail, muteTail).setZero();
	}

	{
		// bottom half of window , top half of input, -> bottom half of output
		const int muteHead = std::clamp(muteFrameCountHead, 0, half);
		const int muteTail = std::clamp(muteFrameCountTail - half, 0, half);
		const int unmuted = half - muteHead - muteTail;

		windowedInput.middleRows(half, muteHead).setZero();

		Window::Apply::special<false>(
			window.segment(window.rows() - muteTail - unmuted, unmuted),
			input.middleRows(input.rows() / 2 - half + muteHead, unmuted),
			windowedInput.middleRows(half + muteHead, unmuted));

		windowedInput.bottomRows(muteTail).setZero();
	}

	scale = window[0];

	return Bungee::log2((int)windowedInput.rows());
}

void Input::checkOverlap(int analysisHop)
{
	if (Instrumentation::threadLocal->enabled)
	{
		const auto rows = windowedInput.rows();
		const auto cols = windowedInput.cols();

		Eigen::ArrayXXf input(rows, cols);
		input.topRows(rows / 2) = windowedInput.bottomRows(rows / 2);
		input.bottomRows(rows / 2) = windowedInput.topRows(rows / 2);

		const auto overlap = input.rows() - std::abs(analysisHop);
		if (overlap > 0 && windowedInputPrevious.rows())
		{
			const auto current = input.middleRows(std::max(0, -analysisHop), overlap);
			const auto previous = windowedInputPrevious.middleRows(std::max(0, analysisHop), overlap);
			const auto error = current * previous;
			const auto metric = error.minCoeff() / (-1e-3f * scale * scale);

			if (metric > 1.f)
				Instrumentation::log("POSSIBLE INPUT PROBLEM: audio correlates poorly with that of previous grain (overlap=%d metric=%f)", overlap, metric);
		}

		windowedInputPrevious = input;
	}
}
} // namespace Bungee
