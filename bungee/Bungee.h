// Copyright (C) 2020-2025 Parabola Research Limited
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <cstdint>

// Underlying C-language API to cross name-mangling domains.
// Necessary, for example, when linking a MinGW-built Bungee DLL into an app built by Visual Studio.

struct Bungee_Request
{
	// Frame-offset within the input audio of the centre-point of this audio grain.
	// NaN signifies an invalid grain that produces no audio output and may be used for flushing.
	double position;

	// Output audio speed. Value of 1 means speed should be unchanged relative to the input audio.
	// Used by Stretcher's internal algorithms only when it's not possible to determine speed by
	// subtracting Request::position of previous grain from Request::position of current grain.
	double speed;

	// Adjustment as a frequency multiplier with a value of 1 meaning no pitch adjustment
	double pitch;

	// Set to have the stretcher forget all previous grains and restart on this grain.
	bool reset;
};

// Information to describe a chunk of  audio required as input
struct Bungee_InputChunk
{
	// Frame offsets relative to the start of the audio track
	int begin, end;
};

// Describes a chunk of audio output
// Output chunks do not overlap and can be appended for seamless playback
struct Bungee_OutputChunk
{
	float *data; // audio output data, not aligned and not interleaved
	int frameCount;
	intptr_t channelStride; // nth audio channel audio starts at data[n * channelStride]

	static constexpr int begin = 0, end = 1;
	Bungee_Request *request[2 /* 0=begin, 1=end */];
};

// Stretcher audio sample rates, in Hz
struct Bungee_SampleRates
{
	int input;
	int output;
};

struct Bungee_Stretcher_FunctionTable
{
	const char *(*version)();
	void *(*create)(Bungee_SampleRates sampleRates, int channelCount, int log2SynthesisHopOverride);
	void (*destroy)(void *implementation);
	int (*maxInputFrameCount)(const void *implementation);
	void (*preroll)(const void *implementation, Bungee_Request *request);
	void (*next)(const void *implementation, Bungee_Request *request);
	Bungee_InputChunk (*specifyGrain)(void *implementation, const Bungee_Request *request);
	void (*analyseGrain)(void *implementation, const float *data, intptr_t channelStride);
	void (*synthesiseGrain)(void *implementation, Bungee_OutputChunk *outputChunk);
	bool (*isFlushed)(const void *implementation);
};

#if defined(bungee_library_EXPORTS)
#	if defined(_WIN32)
#		define BUNGEE_C_API __declspec(dllexport)
#	else
#		define BUNGEE_C_API __attribute__((visibility("default")))
#	endif
#else
#	define BUNGEE_C_API
#endif

extern "C" BUNGEE_C_API Bungee_Stretcher_FunctionTable Bungee_Stretcher_getFunctionTable();
extern "C" BUNGEE_C_API Bungee_Stretcher_FunctionTable BungeePro_Stretcher_getFunctionTable();

#undef BUNGEE_C_API

namespace Bungee {

typedef ::Bungee_Request Request;
typedef ::Bungee_InputChunk InputChunk;
typedef ::Bungee_OutputChunk OutputChunk;
typedef ::Bungee_SampleRates SampleRates;

// Stretcher<Basic> is the open-source implementation contained in this repository
struct Basic;

// Stretcher<Pro> is an enhanced and optimised implementation that is available under commercial license
struct Pro;

template <class Implementation>
struct FunctionTable;

template <>
struct FunctionTable<Basic>
{
	static constexpr auto get = &Bungee_Stretcher_getFunctionTable;
};

template <>
struct FunctionTable<Pro>
{
	static constexpr auto get = &BungeePro_Stretcher_getFunctionTable;
};

template <class Implementation>
struct Stretcher : Bungee_Stretcher_FunctionTable
{
	static const char *version()
	{
		return FunctionTable<Implementation>::get().version();
	}

	inline Stretcher(SampleRates sampleRates, int channelCount, int log2SynthesisHopOverride = 0) :
		Bungee_Stretcher_FunctionTable(FunctionTable<Implementation>::get()),
		p(create(sampleRates, channelCount, log2SynthesisHopOverride))
	{
	}

	inline ~Stretcher()
	{
		destroy(p);
	}

	// Returns the largest number of frames that might be requested by specifyGrain()
	// This helps the caller to allocate large enough buffers because it is guaranteed that
	// InputChunk::frameCount() will not exceed this number.
	inline int maxInputFrameCount() const
	{
		return Bungee_Stretcher_FunctionTable::maxInputFrameCount(p);
	}

	// This function adjusts request.position so that the stretcher has a run in of a few
	// grains before hitting the requested position. Without preroll, the first milliseconds
	// of audio might sound weak or initial transients might be lost.
	inline void preroll(Request &request) const
	{
		Bungee_Stretcher_FunctionTable::preroll(p, &request);
	}

	// This function prepares request.position and request.reset for the subsequent grain.
	// Typically called within a granular loop where playback at constant request.speed is desired.
	inline void next(Request &request) const
	{
		Bungee_Stretcher_FunctionTable::next(p, &request);
	}

	// Specify a grain of audio and compute the necessary segment of input audio.
	// After calling this function, call analyseGrain.
	inline InputChunk specifyGrain(const Request &request)
	{
		return Bungee_Stretcher_FunctionTable::specifyGrain(p, &request);
	}

	// Begins processing the grain. The audio data should correspond to the range
	// specified by specifyGrain's return value. After calling this function, call synthesiseGrain.
	inline void analyseGrain(const float *data, intptr_t channelStride)
	{
		Bungee_Stretcher_FunctionTable::analyseGrain(p, data, channelStride);
	}

	// Complete processing of the grain of audio that was previously set up with calls to specifyGrain and analyseGrain.
	inline void synthesiseGrain(OutputChunk &outputChunk)
	{
		Bungee_Stretcher_FunctionTable::synthesiseGrain(p, &outputChunk);
	}

	// Returns true if every grain in the stretcher's pipeline is invalid (its Request::position was NaN).
	inline bool isFlushed() const
	{
		return Bungee_Stretcher_FunctionTable::isFlushed(p);
	}

	void *const p;
};

} // namespace Bungee
