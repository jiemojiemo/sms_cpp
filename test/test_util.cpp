//
// Created by william on 2020/2/25.
//

#include "sms_util.h"
#include <gmock/gmock.h>
#include <vector>

using namespace std;
using namespace testing;
using namespace sms;


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
    Eigen::ArrayXf test_data_arary(7);
    test_data_arary << 0,1,2,3,4,5,6;
    size_t win_size = test_data_arary.size();

    Eigen::ArrayXf result = SMSUtil::zeroPhaseWindowing(test_data_arary, win_size);

    Eigen::ArrayXf truth(win_size);
    truth << 3,4,5,6,0,1,2;
    ASSERT_THAT(result.size(), Eq(win_size));
    ASSERT_TRUE(ArrayEq(result, truth));
}

TEST(AUnwrap, Plus2PiIfAdjacentValueDiffLargerPi)
{
    vector<float> test_data{3.13, -3.12, 3.12, 3.13, -3.11};
    vector<float> result = SMSUtil::unwrap(test_data.data(), test_data.size());

    ASSERT_THAT(result, ContainerEq(vector<float>({3.13f,3.16318531f,3.12f,3.13f,3.17318531f})));
}