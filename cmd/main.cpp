// Copyright (C) 2020-2025 Parabola Research Limited
// SPDX-License-Identifier: MPL-2.0

#include <bungee/Bungee.h>
#include <bungee/CommandLine.h>

int main(int argc, const char *argv[])
{
	using namespace Bungee;

	Request request{};
#ifndef BUNGEE_EDITION
#	define BUNGEE_EDITION Basic
#endif
	typedef BUNGEE_EDITION Edition;

	static const auto helpString = std::string("Bungee ") + Bungee::Stretcher<Edition>::edition() + " audio speed and pitch changer\n\n" +
		"Version: " + Bungee::Stretcher<Edition>::version() + "\n";

	CommandLine::Options options{"<bungee-command>", helpString};
	CommandLine::Parameters parameters{options, argc, argv, request};
	CommandLine::Processor processor{parameters, request};

	Bungee::Stretcher<Edition> stretcher(processor.sampleRates, processor.channelCount);

	stretcher.enableInstrumentation(parameters["instrumentation"].count() != 0);

	processor.restart(request);
	stretcher.preroll(request);

	const int pushFrameCount = parameters["push"].as<int>();
	if (pushFrameCount)
	{
		// This code exists only to demonstrate the usage of the Bungee stretcher with the Push::InputBuffer
		// See the else part of the code for an example of the native "pull" API.
		std::cout << "Using Push::InputBuffer with " << pushFrameCount << " frames per push\n";

		InputChunk inputChunk = stretcher.specifyGrain(request);

		Push::InputBuffer pushInputBuffer(stretcher.maxInputFrameCount() + pushFrameCount, processor.channelCount);

		pushInputBuffer.grain(inputChunk);

		bool done = false;
		for (int position = 0; !done; position += pushFrameCount)
		{
			// Here we loop over segments of input audio, each with pushFrameCount audio frames.

			// First get pushFrameCount frames of audio from the input
			processor.getInputAudio(pushInputBuffer.inputData(), pushInputBuffer.stride(), position, pushFrameCount);

			// The following function and loop delivers pushFrameCount to Bungee
			// Zero or more output audio chunks will be emitted and we concatenate these.
			pushInputBuffer.deliver(pushFrameCount);
			while (pushInputBuffer.inputFrameCountRequired() <= 0)
			{
				stretcher.analyseGrain(pushInputBuffer.outputData(), pushInputBuffer.stride());

				OutputChunk outputChunk;
				stretcher.synthesiseGrain(outputChunk);

				stretcher.next(request);
				done = processor.write(outputChunk);

				inputChunk = stretcher.specifyGrain(request);
				pushInputBuffer.grain(inputChunk);
			}
		}
	}
	else
	{
		// Regular pull API
		for (bool done = false; !done;)
		{
			InputChunk inputChunk = stretcher.specifyGrain(request);

			const auto muteFrameCountHead = std::max(0, -inputChunk.begin);
			const auto muteFrameCountTail = std::max(0, inputChunk.end - processor.inputFrameCount);

			stretcher.analyseGrain(processor.getInputAudio(inputChunk), processor.inputChannelStride, muteFrameCountHead, muteFrameCountTail);

			OutputChunk outputChunk;
			stretcher.synthesiseGrain(outputChunk);

			stretcher.next(request);

			done = processor.write(outputChunk);
		}
	}

	processor.writeOutputFile();

	return 0;
}
