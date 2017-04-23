#include "test/compression_unittest_base.hpp"
#include "delta_afl_encoding.hpp"
#include "util/statistics/cuda_array_statistics.hpp"
#include "delta_afl_gpu.cuh"
#include <gtest/gtest.h>
#include <boost/bind.hpp>
#include "delta_afl_gpu.cuh"
#include "core/cuda_array.hpp"
#include "string.h"

namespace ddj {

class DeltaAflCompressionTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
	DeltaAflEncoding_Compression_Inst,
	DeltaAflCompressionTest,
    ::testing::Values(1000, 100000));

TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_RandomInts_data)
{
	DeltaAflEncoding encoder;
	std::cout << "TEST" << "\n";
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&DeltaAflEncoding::Encode<int>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<int>, encoder, _1),
			CudaArrayGenerator().GenerateRandomIntDeviceArray(100000, 10, 1000))
	);
}

TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_RandomInts_bigData)
{
	DeltaAflEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&DeltaAflEncoding::Encode<int>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<int>, encoder, _1),
			CudaArrayGenerator().GenerateRandomIntDeviceArray(1000, 10, 1000))
	);
}

TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_RandomFloats_size)
{
	DeltaAflEncoding encoder;
    EXPECT_TRUE(
		TestSize<float>(
			boost::bind(&DeltaAflEncoding::Encode<float>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
    );
}

TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_RandomFloats_data)
{
	DeltaAflEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&DeltaAflEncoding::Encode<float>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<float>, encoder, _1),
			GetFloatRandomData())
	);
}

TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_RandomFloatsWithMaxPrecision2_data)
{
	DeltaAflEncoding encoder;
	EXPECT_TRUE(
		TestContent<float>(
			boost::bind(&DeltaAflEncoding::Encode<float>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<float>, encoder, _1),
			GetFloatRandomDataWithMaxPrecision(2))
	);
}

TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_ConsecutiveInts_size)
{
	DeltaAflEncoding encoder;
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&DeltaAflEncoding::Encode<int>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<int>, encoder, _1),
			GetIntConsecutiveData())
    );
}

TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_ConsecutiveInts_data)
{
	DeltaAflEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&DeltaAflEncoding::Encode<int>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<int>, encoder, _1),
			GetIntConsecutiveData())
	);
}

TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_FakeTime_data)
{
	DeltaAflEncoding encoder;
	EXPECT_TRUE(
		TestContent<time_t>(
			boost::bind(&DeltaAflEncoding::Encode<time_t>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<time_t>, encoder, _1),
			GetFakeDataForTime(0, 0.75, GetSize()))
	);
}

TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_RealTime_data)
{
	DeltaAflEncoding encoder;
	EXPECT_TRUE(
		TestContent<time_t>(
			boost::bind(&DeltaAflEncoding::Encode<time_t>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<time_t>, encoder, _1),
			GetTsIntDataFromTestFile())
	);
}

TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_Short_data)
{
	DeltaAflEncoding encoder;
	EXPECT_TRUE(
		TestContent<short>(
			boost::bind(&DeltaAflEncoding::Encode<short>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<short>, encoder, _1),
			GetFakeDataWithPatternA<short>(0, GetSize()/3, 1, 0, 1e3))
	);
}

TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_Char_data)
{
	DeltaAflEncoding encoder;
	EXPECT_TRUE(
		TestContent<char>(
			boost::bind(&DeltaAflEncoding::Encode<char>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<char>, encoder, _1),
			GetFakeDataWithPatternA<char>(0, GetSize()/3, 1, 0, 1e2))
	);
}

TEST_P(DeltaAflCompressionTest, GetMetadataSize_Consecutive_Int)
{
	TestGetMetadataSize<DeltaAflEncoding, int>(GetIntConsecutiveData());
	TestGetMetadataSize<DeltaAflEncoding, int>(GetIntRandomData(10,100));
	TestGetMetadataSize<DeltaAflEncoding, float>(GetFloatRandomDataWithMaxPrecision(2));
}

TEST_P(DeltaAflCompressionTest, GetCompressedSize_Consecutive_Int)
{
	TestGetCompressedSize<DeltaAflEncoding, int>(GetIntConsecutiveData());
	TestGetCompressedSize<DeltaAflEncoding, int>(GetIntRandomData(10,100));
	TestGetCompressedSize<DeltaAflEncoding, float>(GetFloatRandomDataWithMaxPrecision(2));
}

} /* namespace ddj */
