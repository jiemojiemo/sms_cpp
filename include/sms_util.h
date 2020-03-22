//
// Created by william on 2020/2/25.
//

#pragma once
#include <Eigen/Core>
#include <vector>
#include <tuple>

namespace sms
{
class SMSUtil
{
public:
    static std::vector<float> zeroPhaseWindowing(const float* input, size_t num_input_samples, size_t win_size);
    static Eigen::ArrayXf zeroPhaseWindowing(const Eigen::ArrayXf& input, size_t win_size);

    static std::vector<float> unZeroPhaseWindowing(const float* input, size_t num_input_samples, size_t M);
    static Eigen::ArrayXf unZeroPhaseWindowing(const Eigen::ArrayXf& input, size_t M);

    static std::vector<float> unwrap(const float* input, size_t num_input, float tol = M_PI);
    static Eigen::ArrayXf unwrap(const Eigen::ArrayXf& input, float tol = M_PI);

    /**
     * Peak detection
     * @param input the amplitude of spectrum
     * @param num_input_samples input size
     * @param t threshold(dB)
     * @return index of peaks
     */
    static std::vector<size_t> peakDetection(const float* input, size_t num_input_samples, float t);
    static Eigen::Array<size_t, Eigen::Dynamic, 1> peakDetection(const Eigen::ArrayXf& input, float t);

    /**
     * interpolate peak values using parabolic interpolation
     *
     * @param mx magnitude spectrum
     * @param px phase spectrum
     * @param ploc location of peaks
     * @return (interpolated location, magnitude and phase values)
     */
    static std::tuple<Eigen::ArrayXf, Eigen::ArrayXf, Eigen::ArrayXf> peakInterp(
        const Eigen::ArrayXf& mx,
        const Eigen::ArrayXf& px,
        const Eigen::Array<size_t, Eigen::Dynamic, 1>& ploc);


    /**
     * Linear interpolation
     *
     * @param x the x-coordinate of interpolated value
     * @param fp the y-coordinates of the data points
     * @return interpolated value
     */
    static float LinearInterp(float x, const Eigen::ArrayXf& fp);

    /**
     * 1D array linear interpolation
     *
     * @param xp the 1-d array of x-coordinate of interpolated value
     * @param fp the y-coordinates of the data points
     * @return 1-d array of interpolated value
     */
    static Eigen::ArrayXf LinearInterp(const Eigen::ArrayXf& xp, const Eigen::ArrayXf& fp);
};
}
