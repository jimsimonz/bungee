// Copyright (C) 2020-2025 Parabola Research Limited
// SPDX-License-Identifier: MPL-2.0

#define BUNGEE_BASIC_CPP

#include "Basic.h"
#include "Resample.h"
#include "Synthesis.h"
#include "log2.h"

using namespace Bungee;

namespace Bungee {
extern const char *versionDescription;
}

const char *Bungee_Stretcher_version()
{
	return Bungee::versionDescription;
}

void *Bungee_Stretcher_create(SampleRates sampleRates, int channelCount, int log2SynthesisHop)
{
	return new Basic(sampleRates, channelCount, log2SynthesisHop);
}

void Bungee_Stretcher_destroy(void *implementation)
{
	delete reinterpret_cast<Basic *>(implementation);
}

void Bungee_Stretcher_analyseGrain(void *implementation, const float *data, intptr_t channelStride)
{
	reinterpret_cast<Basic *>(implementation)->analyseGrain(data, channelStride);
}

void Bungee_Stretcher_synthesiseGrain(void *implementation, OutputChunk *outputChunk)
{
	reinterpret_cast<Basic *>(implementation)->synthesiseGrain(*outputChunk);
}

bool Bungee_Stretcher_isFlushed(const void *implementation)
{
	return reinterpret_cast<const Basic *>(implementation)->grains.flushed();
}

Bungee_InputChunk Bungee_Stretcher_specifyGrain(void *implementation, const Request *request)
{
	return reinterpret_cast<Basic *>(implementation)->specifyGrain(*request);
}

int Bungee_Stretcher_maxInputFrameCount(const void *implementation)
{
	return reinterpret_cast<const Basic *>(implementation)->maxInputFrameCount(true);
}

void Bungee_Stretcher_preroll(const void *implementation, Bungee_Request *request)
{
	reinterpret_cast<const Basic *>(implementation)->preroll(*request);
}

void Bungee_Stretcher_next(const void *implementation, Bungee_Request *request)
{
	reinterpret_cast<const Basic *>(implementation)->next(*request);
}

Bungee_Stretcher_FunctionTable Bungee_Stretcher_getFunctionTable()
{
	return Bungee_Stretcher_FunctionTable{
		Bungee_Stretcher_version,
		Bungee_Stretcher_create,
		Bungee_Stretcher_destroy,
		Bungee_Stretcher_maxInputFrameCount,
		Bungee_Stretcher_preroll,
		Bungee_Stretcher_next,
		Bungee_Stretcher_specifyGrain,
		Bungee_Stretcher_analyseGrain,
		Bungee_Stretcher_synthesiseGrain,
		Bungee_Stretcher_isFlushed,
	};
}

namespace Bungee {

Basic::Basic(SampleRates sampleRates, int channelCount, int log2SynthesisHopOverride) :
	Timing(sampleRates, log2SynthesisHopOverride),
	transforms(Fourier::transforms()),
	input(log2SynthesisHop, channelCount, *transforms),
	grains(4),
	output(*transforms, log2SynthesisHop, channelCount, maxOutputFrameCount(true), 0.25f, {1.f, 0.5f})
{
	for (auto &grain : grains.vector)
		grain = std::make_unique<Grain>(log2SynthesisHop, channelCount);
}

InputChunk Basic::specifyGrain(const Request &request)
{
	const Assert::FloatingPointExceptions floatingPointExceptions(0);

	grains.rotate();

	auto &grain = grains[0];
	auto &previous = grains[1];
	return grain.specify(request, previous, sampleRates, log2SynthesisHop);
}

void Basic::analyseGrain(const float *data, std::ptrdiff_t stride)
{
	const Assert::FloatingPointExceptions floatingPointExceptions(FE_INEXACT | FE_UNDERFLOW | FE_DENORMALOPERAND);

	auto &grain = grains[0];
	grain.validBinCount = 0;
	if (grain.valid())
	{
		auto m = grain.inputChunkMap(data, stride);
		auto ref = grain.resampleInput(m, 8 << log2SynthesisHop);

		auto log2TransformLength = input.applyAnalysisWindow(ref);

		transforms->forward(log2TransformLength, input.windowedInput, grain.transformed);

		const auto n = Fourier::binCount(grain.log2TransformLength) - 1;
		grain.validBinCount = std::min<int>(std::ceil(n / grain.resampleOperations.output.ratio), n) + 1;
		grain.transformed.middleRows(grain.validBinCount, n + 1 - grain.validBinCount).setZero();

		grain.log2TransformLength = log2TransformLength;

		for (int i = 0; i < grain.validBinCount; ++i)
		{
			const auto x = grain.transformed.row(i).sum();
			grain.energy[i] = x.real() * x.real() + x.imag() * x.imag();
			grain.phase[i] = Phase::fromRadians(std::arg(x));
		}

		Partials::enumerate(grain.partials, grain.validBinCount, grain.energy);

		if (grain.continuous)
			Partials::suppressTransientPartials(grain.partials, grain.energy, grains[1].energy);
	}
}

void Basic::synthesiseGrain(OutputChunk &outputChunk)
{
	const Assert::FloatingPointExceptions floatingPointExceptions(FE_INEXACT);

	auto &grain = grains[0];
	if (grain.valid())
	{
		auto n = Fourier::binCount(grain.log2TransformLength);

		BUNGEE_ASSERT1(!grain.passthrough || grain.analysis.speed == grain.passthrough);

		Synthesis::synthesise(log2SynthesisHop, grain, grains[1]);

		BUNGEE_ASSERT2(!grain.passthrough || grain.rotation.topRows(grain.validBinCount).isZero());

		auto theta = grain.rotation.topRows(grain.validBinCount).cast<float>() * (std::numbers::pi_v<float> / 0x8000);
		auto t = theta.cos() + theta.sin() * std::complex<float>{0, 1};
		if (grain.reverse())
			grain.transformed.topRows(grain.validBinCount) = grain.transformed.topRows(grain.validBinCount).conjugate().colwise() * t;
		else
			grain.transformed.topRows(grain.validBinCount).colwise() *= t;

		transforms->inverse(grain.log2TransformLength, output.inverseTransformed, grain.transformed);
	}

	output.applySynthesisWindow(log2SynthesisHop, grains, output.synthesisWindow);

	Output::Segment::lapPadding(grains[3].segment, grains[2].segment);

	outputChunk = grains[3].segment.resample(
		output.resampleOffset,
		grains[2].resampleOperations.output,
		grains[1].resampleOperations.output,
		output.bufferResampled);

	outputChunk.request[OutputChunk::begin] = &grains[2].request;
	outputChunk.request[OutputChunk::end] = &grains[1].request;
}

} // namespace Bungee
