//
// Created by william on 2020/2/25.
//

#pragma once
#include <Eigen/Core>
#include <vector>

namespace sms
{
class SMSUtil
{
public:
    static std::vector<float> zeroPhaseWindowing(const float* input, size_t num_input_samples, size_t win_size);

    static Eigen::ArrayXf zeroPhaseWindowing(const Eigen::ArrayXf& input, size_t win_size);

};
}
