//
// Created by william on 2020/3/20.
//

#pragma once
#include <vector>

namespace sms
{
class TestHelper
{
public:
    static std::vector<float> generateSineWave(float freq, float sr, size_t num_sample_to_gen);
};
}
