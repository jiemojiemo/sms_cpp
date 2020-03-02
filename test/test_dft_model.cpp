//
// Created by william on 2020/2/25.
//
#include "libaa/include/dsp/aa_window.h"
#include "libaa/include/dsp/aa_fft.h"
#include "sms_util.h"
#include <gmock/gmock.h>
#include <vector>
#include <numeric>
using namespace testing;
using namespace std;
using namespace libaa;
using namespace Eigen;
using namespace sms;

class DFTModel
{
public:
    static tuple<ArrayXf, ArrayXf> Analyze(const ArrayXf& x, const ArrayXf& w, size_t n_fft)
    {
        ArrayXf norm_w = w / w.sum();
        ArrayXf windowed_w = norm_w * x;
        ArrayXf fft_buffer = SMSUtil::zeroPhaseWindowing(windowed_w, n_fft);

        // fft
        const size_t output_size = n_fft/2 + 1;
        FFT fft(n_fft);
        Eigen::ArrayXcf X(output_size);
        fft.forward(fft_buffer.data(),X.data());

        const float eps = std::numeric_limits<float>::min();
        Eigen::ArrayXf absX = X.abs();
        absX = (absX < eps).select(eps, absX);

        ArrayXf mX = 20 * absX.log10();
        ArrayXf pX = X.arg();

        return std::make_tuple(mX, pX);
    }
};

TEST(ADFTModel, AnalyzeReturnsMagnitudeAndPhaseSpectrum)
{
    size_t input_size = 10;
    size_t n_fft = 16;

    auto w = Window::getWindow(WindowType::kHann, input_size, false);
    Map<ArrayXf> window(w.data(), w.size());
    
    ArrayXf test_data = ArrayXf::Constant(input_size, 1.0f);

    auto [mag, pahse] = DFTModel::Analyze(test_data, window, n_fft);
    size_t output_size = n_fft/2 + 1;

    ASSERT_THAT(mag.size(), Eq(output_size));
    ASSERT_THAT(pahse.size(), Eq(output_size));
}