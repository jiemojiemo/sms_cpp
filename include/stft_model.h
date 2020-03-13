//
// Created by william on 2020/3/12.
//

#pragma once
#include <tuple>
#include <Eigen/Core>
using Eigen::ArrayXf;
using Eigen::ArrayXXf;

namespace sms
{
class STFT
{
public:
    static std::tuple<ArrayXXf, ArrayXXf> analyze(const ArrayXf& x,
                                                  const ArrayXf& w,
                                                  size_t fft_size,
                                                  size_t hop_size);

    static ArrayXf synth(const ArrayXXf& mY,
                         const ArrayXXf& pY,
                         size_t window_size,
                         size_t hop_size);
};
}
