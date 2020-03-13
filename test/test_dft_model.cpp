//
// Created by william on 2020/2/25.
//
#include "dft_model.h"
#include "libaa/include/dsp/aa_window.h"
#include <gmock/gmock.h>

using namespace testing;
using namespace std;
using namespace Eigen;
using namespace sms;
using namespace libaa;


class ADFTModel : public Test
{
public:
    ADFTModel()
    {
        test_data = ArrayXf::Constant(input_size, 1.0f);

        auto w = Window::getWindow(WindowType::kHann, input_size, false);
        window = Map<ArrayXf>(w.data(), w.size());
    }
    size_t input_size = 501;
    size_t n_fft = 1024;
    ArrayXf window;
    ArrayXf test_data;
};

TEST_F(ADFTModel, AnalyzeReturnsMagnitudeAndPhaseSpectrum)
{
    auto [mag, phase] = DFTModel::analyze(test_data, window, n_fft);
    size_t output_size = n_fft/2 + 1;

    ASSERT_THAT(mag.size(), Eq(output_size));
    ASSERT_THAT(phase.size(), Eq(output_size));
}

TEST_F(ADFTModel, Synth)
{
    auto [mag, phase] = DFTModel::analyze(test_data, window, n_fft);

    ArrayXf y = DFTModel::synth(mag, phase, input_size);

    ASSERT_THAT(y.size(), Eq(input_size));
}