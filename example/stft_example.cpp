//
// Created by william on 2020/3/12.
//

#include "libaa/include/fileio/aa_audio_file.h"
#include "stft_model.h"
#include "libaa/include/dsp/aa_window.h"
#include <string>
#include <iostream>
#include <Eigen/Core>
using namespace sms;
using namespace std;
using namespace libaa;
using namespace Eigen;

int main()
{
    string input_path = "../../libaa/res/wav/wav_mono_16bit_44100.wav";
    AudioFile input_file;
    if(input_file.load(input_path) != 0)
    {
        cerr << "open file failed\n";
        return -1;
    }

    const size_t win_size = 1001;
    const size_t fft_size = 1024;
    const size_t hop_size = 256;

    auto num_samples = input_file.getNumFrames();
    float* left_channel = input_file.samples[0].data();
    auto window = Window::getWindow(WindowType::kBlackmanHarris, win_size);

    Map<ArrayXf> x(left_channel, num_samples);
    Map<ArrayXf> w(window.data(), win_size);

    auto [xmX, xpX] = STFT::analyze(x, w, fft_size, hop_size);
    cout << "xmX shape:" << xmX.rows() << "x" << xmX.cols() << endl;

    ArrayXf y = STFT::synth(xmX, xpX, win_size, hop_size);
    input_file.setNumChannles(1);
    input_file.setChannelData(0, y.data(), y.size());
    input_file.saveToWave("istft_result.wav");
}