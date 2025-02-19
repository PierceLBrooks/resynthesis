#!/usr/bin/env python

import sys

from pcm_audio import PcmAudio
from resynthesis import resynthesize


def main():

    if len(sys.argv) < 3:
        sys.exit("Usage: python main.py <input .wav file name> <output .wav file name>")

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    pcm_audio = PcmAudio.from_wave_file(input_filename)
    synthesized_pcm_audio = resynthesize(pcm_audio)
    synthesized_pcm_audio.to_wave_file(output_filename)


if __name__ == "__main__":
    main()
