include_directories(${gmock_SOURCE_DIR}/include)
include_directories(${gtest_SOURCE_DIR}/include)

add_executable(sms_unit_test
        test_util.cpp
        test_dft_model.cpp
        test_stft.cpp
        )

target_link_libraries(sms_unit_test
        gmock_main
        libaa
        sms
        )