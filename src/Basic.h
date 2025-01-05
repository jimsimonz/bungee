// Copyright (C) 2020-2025 Parabola Research Limited
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "Fourier.h"
#include "Grains.h"
#include "Input.h"
#include "Output.h"
#include "Timing.h"

#include <memory>

namespace Bungee {

struct Basic :
	Timing
{
	std::unique_ptr<Fourier::Transforms> transforms;
	Input input;
	Grains grains;
	Output output;

	Basic(SampleRates sampleRates, int channelCount, int log2SynthesisHopOverride);

	InputChunk specifyGrain(const Request &request);

	void analyseGrain(const float *inputAudio, std::ptrdiff_t stride);

	void synthesiseGrain(OutputChunk &outputChunk);

	bool isFlushed() const;
};

} // namespace Bungee
