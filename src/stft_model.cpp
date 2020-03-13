//
// Created by william on 2020/3/12.
//

#include "stft_model.h"
#include "dft_model.h"
#include "sms_util.h"
#include "libaa/include/dsp/aa_window.h"
#include <tuple>


namespace sms
{

std::tuple<ArrayXXf, ArrayXXf> STFT::analyze(const ArrayXf& x,
                                             const ArrayXf& w,
                                             size_t fft_size,
                                             size_t hop_size)
{
    assert(hop_size > 0);

    auto win_size = w.size();
    auto hm1 = (win_size+1)/2;
    auto hm2 = win_size/2;
    ArrayXf append_x( hm2 + x.size() + hm1 );
    append_x << ArrayXf::Zero(hm2), x ,ArrayXf::Zero(hm1); // append zeros at beginning and end

    size_t num_frame = x.size() / hop_size;
    size_t half_fft_size = fft_size/2 + 1;
    ArrayXXf xmX(half_fft_size, num_frame);
    ArrayXXf xpX(half_fft_size, num_frame);

    for(size_t i = 0; i < num_frame; ++i)
    {
        ArrayXf x1 = append_x.segment(i*hop_size, win_size);
        auto [mag, phase] = DFTModel::analyze(x1, w, fft_size);

        xmX.col(i) = mag;
        xpX.col(i) = phase;
    }

    return std::make_tuple(xmX, xpX);
}

ArrayXf STFT::synth(const ArrayXXf& mY,
                    const ArrayXXf& pY,
                    size_t window_size,
                    size_t hop_size)
{
    auto hm1 = (window_size+1)/2;
    auto hm2 = window_size/2;
    auto num_frames = mY.cols();

    ArrayXf y = ArrayXf::Zero(num_frames*hop_size + hm1 + hm2);

    for(size_t i = 0; i < num_frames; ++i)
    {
        ArrayXf y1 = DFTModel::synth(mY.col(i), pY.col(i), window_size);
        y.segment(i*hop_size, window_size) += hop_size * y1;
    }

    return y.segment(hm2, num_frames*hop_size);
}
}