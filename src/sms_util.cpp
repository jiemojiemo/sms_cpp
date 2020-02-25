//
// Created by william on 2020/2/25.
//

#include "include/sms_util.h"
namespace sms
{
std::vector<float> SMSUtil::zeroPhaseWindowing(const float* input, size_t num_input_samples, size_t win_size)
{
    assert(win_size >= num_input_samples);

    std::vector<float> result(win_size, 0.0f);

    const size_t num_left_part = (num_input_samples + 1) / 2;
    const size_t num_right_part = (num_input_samples) / 2;

    size_t input_index = num_input_samples - num_left_part;
    for(int i = 0; i < num_left_part; ++i)
    {
        result[i] = input[input_index++];
    }

    input_index = 0;
    for(int i = win_size - num_right_part; i < win_size; ++i)
    {
        result[i] = input[input_index++];
    }


    return result;
}

Eigen::ArrayXf SMSUtil::zeroPhaseWindowing(const Eigen::ArrayXf& input, size_t win_size)
{
    auto result = zeroPhaseWindowing(input.data(), input.size(), win_size);
    return Eigen::Map<Eigen::ArrayXf>(result.data(), win_size);
}

}