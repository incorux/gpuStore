#include "test/compression_unittest_base.hpp"
#include "delta_afl_encoding.hpp"
#include "util/statistics/cuda_array_statistics.hpp"
#include "delta_afl_gpu.cuh"
#include <gtest/gtest.h>
#include <boost/bind.hpp>
#include "delta_afl_gpu.cuh"
#include "core/cuda_array.hpp"

namespace ddj {

class DeltaAflCompressionTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
	DeltaAflEncoding_Compression_Inst,
	DeltaAflCompressionTest,
    ::testing::Values(1000, 100000));

TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_RandomInts_size)
{
	/*	DeltaAflEncoding encoder;
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&DeltaAflEncoding::Encode<int>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
    );*/
    auto data = GetIntRandomData();
    const int WARP_SIZE = 32;
    unsigned long max_size = data->size() / sizeof(int);
    unsigned int bit_length = CudaArrayStatistics().MinBitCnt<int>(data);
    int cword = sizeof(int) * 8;
    unsigned long data_block_size = cword* WARP_SIZE;
    unsigned long int compressed_data_size = (max_size < data_block_size  ? data_block_size : max_size * sizeof(int));
    compressed_data_size = ((compressed_data_size * bit_length + (data_block_size)-1) / (data_block_size)) * 32 * sizeof(int) + (cword) * sizeof(int);
	int compression_blocks_count = (compressed_data_size + (sizeof(int) * WARP_SIZE) - 1) / (sizeof(int) * WARP_SIZE);
    //
    CudaArray cudaArray;
    std::cout << bit_length << "MinBit\n";
	std::cout << compression_blocks_count << "Blocks number\n";
	std::cout << compressed_data_size << "Compressed data size\n";
	std::cout << data->size() * sizeof(int) << "Data size\n";

	auto compressed_data = CudaPtr<int>::make_shared(compressed_data_size);
	auto dev_data_block_start = CudaPtr<int>::make_shared(compression_blocks_count);
	auto decompressed_data = CudaPtr<int>::make_shared(data->size());
	dev_data_block_start->reset(compression_blocks_count * sizeof(int));
	decompressed_data->reset(data->size() * sizeof(int));

    std::cout << "Compressing...\n";
    run_delta_afl_compress_gpu <int, WARP_SIZE> (bit_length, data->get(), compressed_data->get(), dev_data_block_start->get(), data->size());
    std::cout << "Decompressing...\n";
    run_delta_afl_decompress_gpu <int, WARP_SIZE> (bit_length, compressed_data->get(), dev_data_block_start->get(), decompressed_data->get(), data->size());
	cudaArray.Print(data->copy(100), "encoded");
	cudaArray.Print(dev_data_block_start->copy(compression_blocks_count > 10 ? 10 : compression_blocks_count), "helper");
	cudaArray.Print(decompressed_data->copy(100), "decoded");

    EXPECT_TRUE(CompareDeviceArrays(data->get(), decompressed_data->get(), data->size()));
}
/*
TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_RandomInts_data)
{
	DeltaAflEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&DeltaAflEncoding::Encode<int>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
	);
}

TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_RandomInts_bigData)
{
	DeltaAflEncoding encoder;
	EXPECT_TRUE(
		TestContent<int>(
			boost::bind(&DeltaAflEncoding::Encode<int>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<int>, encoder, _1),
			CudaArrayGenerator().GenerateRandomIntDeviceArray(1<<20, 10, 1000))
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
}*/

} /* namespace ddj */
