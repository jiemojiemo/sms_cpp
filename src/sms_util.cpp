//
// Created by william on 2020/2/25.
//

#include "include/sms_util.h"
using Eigen::ArrayXf;
using Eigen::Map;
using Eigen::Array;

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
// wrap to [-pi,pi]
double angleNorm(double x)
{
    x = fmod(x + M_PI, 2*M_PI);
    if (x < 0)
        x += 2*M_PI;
    return x - M_PI;
}

double phaseUnwrap(double prev, double now)
{
    return prev + angleNorm(now - prev);
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
        result[i] = phaseUnwrap(result[i-1], result[i]);
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
std::vector<size_t> SMSUtil::peakDetection(const float *input, size_t num_input_samples, float t)
{
    Map<const ArrayXf> x(input, num_input_samples);
    auto result_array = peakDetection(x, t);
    std::vector<size_t> result(result_array.data(), result_array.data() + result_array.size());
    return result;
}
Eigen::Array<size_t, Eigen::Dynamic, 1> SMSUtil::peakDetection(const Eigen::ArrayXf &input, float t)
{
    // at least 3 samples
    if(input.size() < 3)
    {
        return Eigen::Array<size_t, Eigen::Dynamic, 1>();
    }

    // x[1:-1]
    Eigen::ArrayXf middle = input.segment(1, input.size() - 2);
    // x[2:]
    Eigen::ArrayXf next = input.segment(2, input.size() - 2);
    // x[0:-2]
    Eigen::ArrayXf prev = input.segment(0, input.size() - 2);


    Eigen::ArrayXi above_threshold = (middle > t).cast<int>();
    Eigen::ArrayXi next_minor = ((middle - next) > 0).cast<int>();
    Eigen::ArrayXi prev_minor = ((middle - prev) > 0).cast<int>();
    Eigen::ArrayXi ploc = above_threshold * next_minor * prev_minor;

    std::vector<size_t> result;
    for(size_t i = 0; i < ploc.size(); ++i)
    {
        if(ploc(i) != 0)
        {
            result.push_back(i + 1); // add 1 to compensate for previous steps
        }
    }

    return Map<Array<size_t, Eigen::Dynamic, 1>>(result.data(), result.size());
}

float SMSUtil::LinearInterp(float x, const Eigen::ArrayXf& fp)
{
    if(x < 0.0)
    {
        return fp(0);
    }

    if(x > static_cast<float>(fp.size()))
    {
        return fp(fp.size() - 1);
    }

    auto prev_index = static_cast<size_t>(std::floorf(x));
    auto next_index = static_cast<size_t>(std::ceilf(x));
    auto fraction = static_cast<float>(next_index) - x;

    return fraction*fp(prev_index) + (1.0f-fraction)*fp(next_index);
}

Eigen::ArrayXf SMSUtil::LinearInterp(const Eigen::ArrayXf &xp, const Eigen::ArrayXf &fp)
{
    ArrayXf result(xp.size());

    for(size_t i = 0; i < xp.size(); ++i)
    {
        result(i) = LinearInterp(xp(i), fp);
    }

    return result;
}
std::tuple<Eigen::ArrayXf, Eigen::ArrayXf, Eigen::ArrayXf> SMSUtil::peakInterp(
    const Eigen::ArrayXf &mx,
    const Eigen::ArrayXf &px,
    const Eigen::Array<size_t, Eigen::Dynamic, 1>& ploc)
{
    ArrayXf val(ploc.size());
    ArrayXf lval(ploc.size());
    ArrayXf rval(ploc.size());
    for(size_t i = 0; i < ploc.size(); ++i)
    {
        val(i) = mx( ploc(i) );
        lval(i) = mx( ploc(i) - 1 );
        rval(i) = mx(ploc(i) + 1);
    }

    ArrayXf ploc_float = ploc.cast<float>();
    ArrayXf interp_ploc = ploc_float + 0.5f*(lval - rval)/(lval - 2.0f*val + rval);
    ArrayXf interp_mag = val - 0.25f*(lval - rval)*(interp_ploc - ploc_float);
    ArrayXf interp_phase = LinearInterp(interp_ploc, px);


    return {interp_ploc, interp_mag, interp_phase};
}


}