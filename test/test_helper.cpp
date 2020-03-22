//
// Created by william on 2020/3/20.
//

#include <vector>
#include <cmath>
#include "test_helper.h"

namespace sms
{

std::vector<float> TestHelper::generateSineWave(float freq, float sr, size_t num_sample_to_gen)
{
    std::vector<float> result(num_sample_to_gen, 0.0f);

    float phase = 0.0f;
    float sr_inverse = 1.0f/sr;
    float phase_step = freq * sr_inverse;

    for(size_t i = 0; i < num_sample_to_gen; ++i)
    {
        result[i] = std::cos(2*M_PI*phase);

        phase += phase_step;
    }

    return result;
}
}
