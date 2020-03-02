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

std::vector<float> SMSUtil::unwrap(const float *input, size_t num_input, float tol)
{
    std::vector<float> result(input, input+num_input);
    if(num_input <= 1)
    {
        return result;
    }

    for(int i = 1; i < num_input; ++i)
    {
        float diff = fabsf(result[i] - result[i-1]);
        if(diff > tol)
        {
            result[i] += 2*M_PI;
        }
    }
    return result;
}

Eigen::ArrayXf SMSUtil::unwrap(const Eigen::ArrayXf& input, float tol)
{
    auto result = unwrap(input.data(), input.size(), tol);
    return Eigen::Map<Eigen::ArrayXf>(result.data(), result.size());
}
std::vector<float> SMSUtil::unZeroPhaseWindowing(const float *input, size_t num_input_samples, size_t win_size)
{
    std::vector<float> result(win_size);
    const size_t num_left_part = (win_size + 1) / 2;
    const size_t num_right_part = (win_size) / 2;

    size_t output_index = 0;
    for(int i = num_input_samples - num_right_part; i < num_input_samples; ++i)
    {
        result[output_index++] = input[i];
    }
    for(int i = 0; i < num_left_part; ++i)
    {
        result[output_index++] = input[i];
    }

    return result;
}

Eigen::ArrayXf SMSUtil::unZeroPhaseWindowing(const Eigen::ArrayXf& input, size_t win_size)
{
    auto result = unZeroPhaseWindowing(input.data(), input.size(), win_size);
    return Eigen::Map<Eigen::ArrayXf>(result.data(), result.size());
}

}