#include <gtest/gtest.h>

#include "bn_fin.hpp"
#include "fin.hpp"
#include "conv_fin.hpp"

// to use the template class
using namespace fin;
// using json = nlohmann::json;

// define unit tests
namespace TestType {
const int TestApplicability = 1;
const int GetSolverList     = 2;
}

// To use a test fixture, derive a class from testing::Test.
class bn_ut_test : public testing::Test
{
    using testing::Test::SetUp;

    protected:
    void SetUp(json* jObj, int type)
    {
        std::string testSamplePath = TEST_RESOURCE_DIR;

        switch(type)
        {
        case TestType::TestApplicability:
        {
            if(!boost::filesystem::exists(testSamplePath + "bn_configs_fin_all.json"))
            {
                std::cerr << "File: BN config file:bn_configs_fin_all.json doesn't exist"
                          << std::endl;
                {
                    exit(-1);
                }
            }

            boost::filesystem::path input_filename(testSamplePath + "bn_configs_fin_all.json");
            std::ifstream input_file(input_filename.string());
            if(!input_file)
            {
                throw std::runtime_error("Error loading bn_configs_fin_all.json file: " +
                                         input_filename.string());
            }
            input_file >> *jObj;
        }
        break;
        case TestType::GetSolverList:
        {
            if(!boost::filesystem::exists(testSamplePath + "bn_fin_solvers_only.json"))
            {
                std::cerr << "File: BN config file:bn_solvers_only.json doesn't exist" << std::endl;
                {
                    exit(-1);
                }
            }
            boost::filesystem::path input_filename(testSamplePath + "bn_fin_solvers_only.json");
            std::ifstream input_file(input_filename.string());
            if(!input_file)
            {
                throw std::runtime_error("Error loading bn_solvers_only.json file: " +
                                         input_filename.string());
            }
            input_file >> *jObj;
        }
        break;
        default: std::cerr << "No matching Test type." << std::endl; break;
        }
    }
};

TEST_F(bn_ut_test, applicability_test)
{

    json jObj;
    SetUp(&jObj, TestType::TestApplicability);

    std::unique_ptr<fin::BNFin<float, float>> f    = nullptr;
    std::unique_ptr<fin::BNFin<float16, float>> f1 = nullptr;

    for(auto& it : jObj)
    {
        auto command = it;
        if(command.contains("config"))
        {
            if(command["config"]["cmd"] == "bnorm")
            {
                f         = std::make_unique<fin::BNFin<float, float>>(command);
                EXPECT_TRUE(f != nullptr);
                int value = f->TestApplicability();
                ASSERT_EQ(value, 0);
            }
            else if(command["config"]["cmd"] == "bnormfp16")
            {
                f1        = std::make_unique<fin::BNFin<float16, float>>(command);
                EXPECT_TRUE(f1 != nullptr);
                int value = f1->TestApplicability();
                ASSERT_EQ(value, 0);
            }
        }
    }
}

TEST_F(bn_ut_test, get_solver_list)
{
    json jObj;
    SetUp(&jObj,TestType::GetSolverList);

    std::unique_ptr<fin::BaseFin> f    = nullptr;
    for(auto& it : jObj)
    {
        auto command = it;
        for(auto& step_it : command["steps"])
        {
            if(step_it == "get_solvers")
            {
                f = std::make_unique<fin::ConvFin<float, float>>();
                EXPECT_TRUE(f != nullptr);
                int result = f->GetSolverList();
                ASSERT_EQ(result, 0);
            }
        }
    }
}
