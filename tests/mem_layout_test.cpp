#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <nlohmann/json.hpp>
#include <string>
#include <fstream>

#include <conv_fin.hpp>

using json = nlohmann::json;

TEST(MemoryLayoutTest, BasicMemLayout)
{
    std::string input_filename = TEST_RESOURCE_DIR "fin_input_find_compile2.json";
    std::ifstream input_file(input_filename);
    if(!input_file)
    {
        EXPECT_FALSE(true) << "ERROR: cannot open test file " << input_filename << std::endl;
    }

    json j;
    input_file >> j;
    input_file.close();
    for(auto& it : j)
    {
        auto command = it;
        if(command["config"]["cmd"] == "conv")
        {
            fin::ConvFin<float, float> tmp(command);
            ASSERT_TRUE(tmp.inputTensor.desc.GetLayout_t() ==
                        miopenTensorLayout_t::miopenTensorNCHW);
            // set the layout from json file
            tmp.GetandSetData();
            ASSERT_TRUE(tmp.inputTensor.desc.GetLayout_t() ==
                        miopenTensorLayout_t::miopenTensorNHWC);
            ASSERT_TRUE(tmp.inputTensor.desc.GetLayout_t() !=
                        miopenTensorLayout_t::miopenTensorNCHW);
        }
    }
}
