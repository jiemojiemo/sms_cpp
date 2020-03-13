//
// Created by william on 2020/3/12.
//


#include "stft_model.h"
#include "libaa/include/dsp/aa_window.h"
#include <gmock/gmock.h>
#include <iostream>
using namespace testing;
using namespace sms;
using namespace libaa;
using namespace std;


class ASTFT : public Test
{
public:
    void SetUp() override
    {
        test_data = ArrayXf::Constant(4096, 1.0f);
        auto w = Window::getWindow(WindowType::kHann, window_size, false);
        window = Eigen::Map<ArrayXf>(w.data(), w.size());
    }

    const size_t num_samples = 4096;
    const size_t window_size = 501;
    const size_t fft_size = 1024;
    const size_t hop_size = 256;
    ArrayXf test_data;
    ArrayXf window;
};

TEST_F(ASTFT, analyze)
{
    auto [xmX, xpX] = STFT::analyze(test_data, window, fft_size, hop_size);

    ASSERT_THAT(xmX.rows(), Eq(fft_size/2 + 1));
    ASSERT_THAT(xmX.cols(), Eq((num_samples)/hop_size));
    ASSERT_THAT(xpX.rows(), Eq(fft_size/2 + 1));
    ASSERT_THAT(xpX.cols(), Eq((num_samples)/hop_size));
}

TEST_F(ASTFT, synth)
{
    auto [xmX, xpX] = STFT::analyze(test_data, window, fft_size, hop_size);
    auto y = STFT::synth(xmX, xpX, window_size, hop_size);
    cout << y << endl;
    ASSERT_THAT(y.size(), Eq(test_data.size()));
}