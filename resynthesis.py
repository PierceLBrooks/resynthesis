import sys
import os.path
from io import StringIO

import numpy

from algorithm import GeneticAlgorithm, Population
from pcm_audio import PcmAudio
from sound import get_sound_factory
from spectrogram import Spectrogram
from PIL import Image

def get_distance(left, right):
    distance = 0.0
    for i in range(min(3, min(len(left), len(right)))):
        distance += (left[i] - right[i]) ** 2.0
    return distance ** 0.5

def get_comparison(left, right):
    similarity = 0.0
    left_image = Image.open(left)
    right_image = Image.open(right)
    left_size = left_image.size
    right_size = right_image.size
    left_pixels = left_image.load()
    right_pixels = right_image.load()
    width = min(left_size[0], right_size[0])
    height = min(left_size[1], right_size[1])
    for x in range(width):
        for y in range(height):
            similarity += get_distance(list(left_pixels[x, y]), list(right_pixels[x, y])) / 441.673
    similarity /= float(width) * float(height)
    similarity = 1.0 - min(1.0, max(0.0, similarity))
    similarity *= 100.0
    return int(similarity)

def resynthesize(reference_pcm_audio):
    Spectrogram(reference_pcm_audio.samples).to_tga_file(
        "reference_spectrogram.tga")

    algorithm = GeneticAlgorithm()

    best_score = None
    pcm_audio = None
    sounds = []

    if os.path.exists("base.wav"):
        print("Using 'base.wav' as a base sound for additive sythesis")
        pcm_audio = PcmAudio.from_wave_file("base.wav")

    generations = 20
    populations = 80
    if len(sys.argv) > 3:
        generations = int(sys.argv[3])
    if len(sys.argv) > 4:
        populations = int(sys.argv[4])
    for generation in range(generations):
        print(str(generation) + " / " + str(generations) + " @ " + str(populations))
        genome_factory = get_sound_factory(reference_pcm_audio, pcm_audio)
        population = Population(genome_factory, populations)
        best_sound = algorithm.run(population)

        if best_score is not None and best_sound.score < best_score:
            print("The algorithm failed to produce a better sound on this step")
            break

        print(best_sound)
        pcm_audio = best_sound.to_pcm_audio()
        pcm_audio.to_wave_file("debug%d.wav" % generation)

        Spectrogram(pcm_audio.samples).to_tga_file("out.tga")
        print("reference_spectrogram.tga - out.tga = " + str(get_comparison(os.path.join(os.getcwd(), "reference_spectrogram.tga"), os.path.join(os.getcwd(), "out.tga"))) + "%")

        best_score = best_sound.score

        sounds.append(best_sound)

    construct_csound_file(sounds, pcm_audio)

    return pcm_audio


def construct_csound_file(sounds, pcm_audio, filename="out.csd"):
    signed_short_max = 2**15 - 1

    sound_duration = pcm_audio.duration
    sound_amplitude = pcm_audio.samples.max() / signed_short_max

    # We assume that a frequency of a partial with the maximal energy is a base
    # sound frequency

    def energy(sound):
        sound._base_pcm_audio = None
        return numpy.abs(sound.to_pcm_audio().samples).sum()

    sounds = sorted(sounds, key=energy, reverse=True)

    sound_frequency = sounds[0]._frequency

    # Construct instrument code
    code_stream = StringIO()

    print("aResult = 0", file=code_stream)

    for sound in sounds:
        signal = (
            "aSignal oscils {amplitude}, iFrequency * {freqency_ratio:.2f}, "
            "{normalized_phase:.2f}"
            .format(
                amplitude=1,
                freqency_ratio=sound._frequency / sound_frequency,
                normalized_phase=sound._phase / (2 * numpy.pi)
            )
        )
        print(signal, file=code_stream)

        sound._sort_amplitude_envelope_points()

        times = [point.time for point in sound._amplitude_envelope_points]
        normalized_amplitudes = [
            (point.value / signed_short_max) / sound_amplitude for point in
                sound._amplitude_envelope_points]

        envelope = "aEnvelope linseg iAmplitude * {amplitude:.3f}".format(
            amplitude=normalized_amplitudes[0])

        previous_point_time = 0
        for time, normalized_amplitude in zip(times, normalized_amplitudes):
            envelope = (
                "{envelope}, "
                "iDuration * {duration:.3f}, "
                "iAmplitude * {amplitude:.3f}"
                .format(
                    envelope=envelope,
                    duration=time / sound_duration - previous_point_time,
                    amplitude=normalized_amplitude
                )
            )
            previous_point_time = time / sound_duration
        envelope = "{envelope}, 0, 0".format(envelope=envelope)
        print(envelope, file=code_stream)

        print("aResult = aResult + aSignal * aEnvelope", file=code_stream)

    print("out aResult", file=code_stream)

    instrument_code = code_stream.getvalue()

    code_stream.close()

    # Construct score
    score = "i 1 0 {duration:.2f} {amplitude:.2f} {frequency:.1f}".format(
        duration=sound_duration,
        amplitude=sound_amplitude,
        frequency=sound_frequency)

    with open("template.csd", "r") as template_file:
        template = template_file.read()

    output = template
    output = output.replace("; %instrument_code%", instrument_code)
    output = output.replace("; %score%", score)

    with open(filename, "w") as output_file:
        print(output, file=output_file)
