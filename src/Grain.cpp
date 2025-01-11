// Copyright (C) 2020-2025 Parabola Research Limited
// SPDX-License-Identifier: MPL-2.0

#include "Grain.h"
#include "Fourier.h"
#include "Instrumentation.h"

#include "bungee/Bungee.h"

#include <limits>

namespace Bungee {

using namespace Internal;

Grain::Grain(int log2SynthesisHop, int channelCount) :
	log2TransformLength(log2SynthesisHop + 3),
	segment(log2SynthesisHop, channelCount),
	inputResampled(1 << log2TransformLength, channelCount)
{
	request.position = request.speed = std::numeric_limits<float>::quiet_NaN();
	request.pitch = 1.;
	Fourier::resize<true>(log2TransformLength, channelCount, transformed);
	Fourier::resize<true>(log2TransformLength, 1, phase);
	Fourier::resize<true>(log2TransformLength, 1, energy);
	Fourier::resize<true>(log2TransformLength, 1, rotation);
	Fourier::resize<true>(log2TransformLength, 1, delta);
	partials.reserve(1 << log2TransformLength);
}

InputChunk Grain::specify(const Request &r, Grain &previous, SampleRates sampleRates, int log2SynthesisHop, double bufferStartPosition)
{
	request = r;
	BUNGEE_ASSERT1(request.pitch > 0.);

	const Assert::FloatingPointExceptions floatingPointExceptions(FE_INEXACT);

	const auto unitHop = (1 << log2SynthesisHop) * resampleOperations.setup(sampleRates, request.pitch);

	requestHop = request.position - previous.request.position;

	if (!std::isnan(request.speed) && !std::isnan(requestHop) && std::abs(request.speed * unitHop - requestHop) > 1.)
		Instrumentation::log("specifyGrain: speed=%f implies hop of %f/%f but position has advanced by %f/%f since previous grain", request.speed, request.speed * unitHop, sampleRates.input, requestHop, sampleRates.input);

	if (std::isnan(requestHop) || request.reset)
		requestHop = request.speed * unitHop;

	analysis.hopIdeal = requestHop * resampleOperations.input.ratio;

	continuous = !request.reset && !std::isnan(previous.request.position);
	if (continuous)
	{
		analysis.positionError = previous.analysis.positionError - analysis.hopIdeal;
		analysis.hop = std::round(-analysis.positionError);
		analysis.positionError += analysis.hop;
	}
	else
	{
		analysis.hop = std::round(analysis.hopIdeal);
		analysis.positionError = std::round(request.position) - request.position;
	}

	analysis.speed = analysis.hopIdeal / (1 << log2SynthesisHop);

	{
		passthrough = std::abs(analysis.speed) == 1. ? int(analysis.speed) : 0;
		if (continuous && passthrough != previous.passthrough)
			passthrough = 0;
	}

	log2TransformLength = log2SynthesisHop + 3;
	inputResampled.frameCount = 1 << log2TransformLength;

	{
		auto halfInputFrameCount = inputResampled.frameCount / 2;
		if (resampleOperations.input.ratio != 1.f)
			halfInputFrameCount = int(std::round(halfInputFrameCount / resampleOperations.input.ratio)) + 1;
		inputChunk.begin = int(std::round(request.position - bufferStartPosition)) - halfInputFrameCount;
		inputChunk.end = int(std::round(request.position - bufferStartPosition)) + halfInputFrameCount;

		return inputChunk;
	}
}

Eigen::Ref<Eigen::ArrayXXf> Grain::resampleInput(Eigen::Ref<Eigen::ArrayXXf> input, int log2WindowLength, int &muteFrameCountHead, int &muteFrameCountTail)
{
	if (resampleOperations.input.function)
	{
		float offset = float(inputChunk.begin - request.position);
		offset *= resampleOperations.input.ratio;
		offset += 1 << (log2WindowLength - 1);
		offset -= analysis.positionError;

		resampleOperations.input.function(inputResampled, offset, input, resampleOperations.input.ratio, resampleOperations.input.ratio, false, muteFrameCountHead, muteFrameCountTail);

		muteFrameCountHead = muteFrameCountTail = 0;

		return inputResampled.unpadded().topRows(inputResampled.frameCount);
	}
	else
	{
		return input;
	}
}
} // namespace Bungee
