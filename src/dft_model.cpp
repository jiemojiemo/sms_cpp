//
// Created by william on 2020/3/2.
//

#include "dft_model.h"
#include "libaa/include/dsp/aa_window.h"
#include "libaa/include/dsp/aa_fft.h"
#include "sms_util.h"
#include <tuple>
#include <numeric>
#include <complex>

namespace sms
{
using libaa::FFT;
using namespace std;

std::tuple<ArrayXf, ArrayXf> DFTModel::analyze(const ArrayXf& x, const ArrayXf& w, size_t n_fft)
{
    ArrayXf norm_w = w / w.sum();
    ArrayXf windowed_w = norm_w * x;
    ArrayXf fft_buffer = SMSUtil::zeroPhaseWindowing(windowed_w, n_fft);

    // fft
    const size_t output_size = n_fft/2 + 1;
    FFT fft(n_fft);
    Eigen::ArrayXcf X(output_size);
    fft.forward(fft_buffer.data(),X.data());

    // avoid log zero
    const float eps = std::numeric_limits<float>::min();
    Eigen::ArrayXf absX = X.abs();
    absX = (absX < eps).select(eps, absX);

    ArrayXf mX = 20 * absX.log10();
    ArrayXf pX = X.arg();

    pX = SMSUtil::unwrap(pX);

    return std::make_tuple(mX, pX);
}

Eigen::ArrayXf DFTModel::synth(const ArrayXf& mX, const ArrayXf& pX, size_t win_size)
{
    assert(mX.size() == pX.size());

    size_t n_fft = (mX.size() - 1) * 2;
    ArrayXcf Y = Eigen::pow(10, (mX/20.0f)) * Eigen::exp(1if * pX);

    FFT fft(n_fft);
    ArrayXf fft_buffer(n_fft);
    fft.inverse(Y.data(), fft_buffer.data());
    // inverse fft normalization
    fft_buffer = fft_buffer / static_cast<float>(n_fft);

    ArrayXf y = SMSUtil::unZeroPhaseWindowing(fft_buffer, win_size);
    return y;
}
}