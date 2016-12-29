#include "test/compression_unittest_base.hpp"
#include "delta_afl_encoding.hpp"
#include "util/statistics/cuda_array_statistics.hpp"
#include "delta_afl_gpu.cuh"
#include <gtest/gtest.h>
#include <boost/bind.hpp>
#include "delta_afl_gpu.cuh"
#include "core/cuda_array.hpp"
#include "string.h"

#define LNBITSTOMASK(n) ((1L<<(n)) - 1)

namespace ddj {

class DeltaAflCompressionTest : public CompressionUnittestBase {};

INSTANTIATE_TEST_CASE_P(
	DeltaAflEncoding_Compression_Inst,
	DeltaAflCompressionTest,
    ::testing::Values(1000, 100000));

TEST_P(DeltaAflCompressionTest, DeltaAfl_Encode_Decode_RandomInts_size)
{
	// Variable processing
    const int WARP_SIZE = 32;
    unsigned int bit_length = 10; //CudaArrayStatistics().MinBitCnt<int>(data); // Returning 10 for ints
    int cword = sizeof(int) * 8;
    unsigned long max_size = 1000;
    unsigned long data_size =  max_size * sizeof(int);
    unsigned long data_chunk = cword * WARP_SIZE;
    unsigned long compressed_data_size = (max_size < data_chunk  ? data_chunk : max_size);
    compressed_data_size = ((compressed_data_size * bit_length + (data_chunk)-1) / (data_chunk)) * 32 * sizeof(int) + (cword) * sizeof(int);
	int compression_blocks_count = (compressed_data_size + (sizeof(int) * WARP_SIZE) - 1) / (sizeof(int) * WARP_SIZE);
    //
    CudaArray cudaArray;
    std::cout << bit_length << "MinBit\n";
	std::cout << compression_blocks_count << "Blocks number\n";
	std::cout << compressed_data_size << "Compressed data size\n";
	std::cout << data_size << "Data size\n";

	// Declarations and instantiations
    auto result = CudaPtr<int>::make_shared(data_size);
    auto initial_data = CudaPtr<int>::make_shared(data_size);
    auto data_block_start = CudaPtr<int>::make_shared(compression_blocks_count);
    auto compressed_data = CudaPtr<int>::make_shared(compressed_data_size);

    // Create input data
    std::cout << "Before compression host data:\n";
    //COPYPASTA
    srand (time(NULL));
    __xorshf96_x=(unsigned long) rand();
    __xorshf96_y=(unsigned long) rand();
    __xorshf96_z=(unsigned long) rand();
    unsigned long mask = LNBITSTOMASK(bit_length);
    std::vector<int> host_data;
	for (unsigned long i = 0; i < max_size; i++){
		host_data.push_back(max_size - i); //xorshf96() & mask;
		if(i < 100)
			printf("%i ", host_data[i]);
	}
	initial_data->fillFromHost(host_data.data(), data_size);
	//COPYPASTA END
    std::cout << "\n";
    std::cout << "Compressing...\n";
    run_delta_afl_compress_gpu <int, WARP_SIZE> (bit_length, initial_data->get(), compressed_data->get(), data_block_start->get(), max_size);
    std::cout << "Decompressing...\n";
    run_delta_afl_decompress_gpu <int, WARP_SIZE> (bit_length, compressed_data->get(), data_block_start->get(), result->get(), max_size);
    std::cout << "Decompressed";

    int* host_data2;
    host_data2 = result->copyToHost()->data();
    for(int i = 0; i < 100; i++)
    	printf("%i ", host_data2[i]);

    EXPECT_TRUE(CompareDeviceArrays(initial_data->get(), result->get(), max_size));
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
