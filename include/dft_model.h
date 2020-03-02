//
// Created by william on 2020/3/2.
//

#pragma once

#include "sms_util.h"
#include <tuple>

using Eigen::ArrayXf;
using Eigen::ArrayXcf;

namespace sms
{
class DFTModel
{
public:
    static std::tuple<ArrayXf, ArrayXf> analyze(const ArrayXf& x, const ArrayXf& w, size_t n_fft);

    static ArrayXf synth(const ArrayXf& mX, const ArrayXf& pX, size_t win_size);
};
}
