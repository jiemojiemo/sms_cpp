//
// Created by william on 2020/2/25.
//

#include "libaa/include/dsp/aa_window.h"
#include "sms_util.h"
#include "test_helper.h"
#include "dft_model.h"
#include <gmock/gmock.h>
#include <vector>

using namespace std;
using namespace testing;
using namespace sms;
using namespace Eigen;
using namespace libaa;

class AZeroPhaseWindowing : public Test
{
public:
    vector<float> test_data{0,1,2,3,4,5,6};
};

bool ArrayEq(const Eigen::ArrayXf& lhs, const Eigen::ArrayXf& rhs)
{
    if(lhs.size() != rhs.size())
    {
        return false;
    }

    for(int i = 0; i < lhs.size(); ++i)
    {
        if( std::fabs(lhs(i) - rhs(i)) > 1e-7)
            return false;
    }

    return true;
}


TEST_F(AZeroPhaseWindowing, ReturnWinSizeResult)
{
    size_t win_size = test_data.size();

    vector<float> result = SMSUtil::zeroPhaseWindowing(test_data.data(), test_data.size(), win_size);

    ASSERT_THAT(result.size(), Eq(win_size));
}

TEST_F(AZeroPhaseWindowing, ChangesLeftAndRightPart)
{
    size_t win_size = test_data.size();

    vector<float> result = SMSUtil::zeroPhaseWindowing(test_data.data(), test_data.size(), win_size);

    ASSERT_THAT(result, ContainerEq(vector<float>({3,4,5,6,0,1,2})));
}

TEST_F(AZeroPhaseWindowing, PaddingZeroInMiddleIfWindowSizeLargerThanInputSize)
{
    size_t win_size = test_data.size() + 2;

    vector<float> result = SMSUtil::zeroPhaseWindowing(test_data.data(), test_data.size(), win_size);

    ASSERT_THAT(result, ContainerEq(vector<float>({3,4,5,6,0,0,0,1,2})));
}

TEST_F(AZeroPhaseWindowing, ReturnsArrayIfInputIsEigenArray)
{
    Eigen::ArrayXf test_data_arary = Map<ArrayXf>(test_data.data(), test_data.size());
    size_t win_size = test_data_arary.size();

    Eigen::ArrayXf result = SMSUtil::zeroPhaseWindowing(test_data_arary, win_size);

    Eigen::ArrayXf truth(win_size);
    truth << 3,4,5,6,0,1,2;
    ASSERT_THAT(result.size(), Eq(win_size));
    ASSERT_TRUE(ArrayEq(result, truth));
}

TEST_F(AZeroPhaseWindowing, UnZeroPhasingRestore)
{
    size_t input_size = test_data.size();
    size_t win_size = input_size + 2;
    vector<float> zero_phase_result = SMSUtil::zeroPhaseWindowing(test_data.data(), test_data.size(), win_size);
    vector<float> unzero_phase_result = SMSUtil::unZeroPhaseWindowing(zero_phase_result.data(), zero_phase_result.size(), input_size);

    ASSERT_THAT(unzero_phase_result.size(), Eq(input_size));
    ASSERT_THAT(unzero_phase_result, ContainerEq(test_data));
}

TEST_F(AZeroPhaseWindowing, UnZeroPhasingRestoreEigenArray)
{
    Eigen::ArrayXf test_data_arary = Map<ArrayXf>(test_data.data(), test_data.size());

    size_t input_size = test_data_arary.size();
    size_t win_size = input_size + 2;
    ArrayXf zero_phase_result = SMSUtil::zeroPhaseWindowing(test_data_arary, win_size);
    ArrayXf unzero_phase_result = SMSUtil::unZeroPhaseWindowing(zero_phase_result, input_size);

    ASSERT_THAT(unzero_phase_result.size(), Eq(input_size));
    ASSERT_TRUE(ArrayEq(unzero_phase_result, test_data_arary));
}

class AUnwrap : public Test
{
public:
    vector<float> test_data{3.13, -3.12, 3.12, 3.13, -3.11};
    vector<float> ground_truth{3.13f,3.16318531f,3.12f,3.13f,3.17318531f};
};

TEST_F(AUnwrap, Plus2PiIfAdjacentValueDiffLargerPi)
{
    vector<float> result = SMSUtil::unwrap(test_data.data(), test_data.size());

    ASSERT_THAT(result, ContainerEq(ground_truth));
}

TEST_F(AUnwrap, ReturnsArrayIfInputIsEigenArray)
{
    Map<ArrayXf> test_array(test_data.data(), test_data.size());
    Map<ArrayXf> ground_truth_array(ground_truth.data(), ground_truth.size());

    auto result = SMSUtil::unwrap(test_array);
    ASSERT_TRUE(ArrayEq(result, ground_truth_array));
}

class APeakDetection : public Test
{
public:
    void SetUp() override
    {
        auto window = Window::getWindow(WindowType::kHann, window_size, false);
        auto sine_440 = TestHelper::generateSineWave(freq, sr, window_size);

        w = Map<ArrayXf>(window.data(), window.size());
        x = Map<ArrayXf>(sine_440.data(), sine_440.size());
    }

    size_t window_size = 501;
    size_t fft_size = 512;
    float sr = 44100;
    float freq = 440;
    float freq_resolution = sr/fft_size;
    ArrayXf w;
    ArrayXf x;
};

TEST_F(APeakDetection, ReturnsEmptyLocationIfInputLessThan3Samples)
{
    vector<float> test_data(2);
    float threshold = 0;

    auto result = SMSUtil::peakDetection(test_data.data(), test_data.size(), threshold);

    ASSERT_THAT(result.size(), Eq(0));
}

TEST_F(APeakDetection, GetPeakLocationFromMagnitudes)
{
    auto [mx, px] = DFTModel::analyze(x,w,fft_size);
    float threshold = -40.0f;

    auto result = SMSUtil::peakDetection(mx.data(), mx.size(), threshold);

    ASSERT_THAT(result.size(), Eq(1));
    ASSERT_THAT(result[0], Eq(static_cast<size_t>(freq/freq_resolution)));
}

TEST_F(APeakDetection, InterpolatePeak)
{
    auto [mx, px] = DFTModel::analyze(x,w,fft_size);
    float threshold = -40.0f;
    auto result = SMSUtil::peakDetection(mx, threshold);

    auto [iploc, ipmag, ipphase] = SMSUtil::peakInterp(mx, px, result);

    ASSERT_THAT(iploc(0)*freq_resolution, FloatNear(freq, 2.0f));
}

class ALinearInterpolation : public Test
{
public:
    ALinearInterpolation() :
        fp(num_fp),
        xp(num_xp)
    {

    }

    void SetUp() override
    {
        fp << 3,2,0;
        xp << 0.5, 1.5;
    }
    size_t num_fp = 3;
    size_t num_xp = 2;
    ArrayXf fp;
    ArrayXf xp;

};

TEST_F(ALinearInterpolation, ReturnsFirstValueIfXcoordinateLessThanZero)
{
    float x = -1.0;

    float interp_x = SMSUtil::LinearInterp(x, fp);
    ASSERT_THAT(interp_x, Eq(fp(0)));
}

TEST_F(ALinearInterpolation, ReturnsLastValueIfXcoordinateGreaterThanFPsize)
{
    float x = 4;
    float interp_x = SMSUtil::LinearInterp(x, fp);
    ASSERT_THAT(interp_x, Eq(fp(fp.size() -1)));
}

TEST_F(ALinearInterpolation, ReturnsLinearInterpValue)
{
    float x = 1.5;

    float interp_x = SMSUtil::LinearInterp(x, fp);
    ASSERT_THAT(interp_x, FloatEq(1.0f));
}

TEST_F(ALinearInterpolation, Interpolation1DArray) {
    ArrayXf inter_x = SMSUtil::LinearInterp(xp, fp);

    ASSERT_THAT(inter_x.size(), Eq(xp.size()));
    ASSERT_THAT(inter_x(0), FloatEq((3.0f + 2.0f) / 2));
    ASSERT_THAT(inter_x(1), FloatEq(1.0f));
}