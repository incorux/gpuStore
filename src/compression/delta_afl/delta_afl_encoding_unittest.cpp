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
	/*	DeltaAflEncoding encoder;
    EXPECT_TRUE(
		TestSize<int>(
			boost::bind(&DeltaAflEncoding::Encode<int>, encoder, _1),
			boost::bind(&DeltaAflEncoding::Decode<int>, encoder, _1),
			GetIntRandomData())
    );*/
    auto data = GetIntRandomData();
    const int WARP_SIZE = 32;
    unsigned int bit_length = CudaArrayStatistics().MinBitCnt<int>(data);
    int cword = sizeof(int) * 8;
    unsigned long max_size = data->size();
    unsigned long data_size =  max_size * sizeof(int);
    unsigned long data_chunk = cword* WARP_SIZE;
    unsigned long compressed_data_size = (max_size < data_chunk  ? data_chunk : max_size);
    compressed_data_size = ((compressed_data_size * bit_length + (data_chunk)-1) / (data_chunk)) * 32 * sizeof(int) + (cword) * sizeof(int);
	int compression_blocks_count = (compressed_data_size + (sizeof(int) * WARP_SIZE) - 1) / (sizeof(int) * WARP_SIZE);
    //
    CudaArray cudaArray;
    std::cout << bit_length << "MinBit\n";
	std::cout << compression_blocks_count << "Blocks number\n";
	std::cout << compressed_data_size << "Compressed data size\n";
	std::cout << data_size << "Data size\n";

	__device__ int *dev_out;
	__device__ int *dev_data;
	__device__ int *dev_data_block_start;
    int *host_data;
    int *host_data2;
    int *host_data_block_start;

    cudaMallocHost((void**)&host_data,  data_size);
    cudaMallocHost((void**)&host_data2,  data_size);
    cudaMallocHost((void**)&host_data_block_start,  compression_blocks_count * sizeof(unsigned long));

    cudaMalloc((void **) &dev_out, compressed_data_size);
    cudaMalloc((void **) &dev_data, data_size);
    cudaMalloc((void **) &dev_data_block_start, compression_blocks_count * sizeof(unsigned long));

    //host_data = &data->copyToHost();
    std::cout << "Before compression host data:\n";
    //COPYPASTA
    srand (time(NULL));
    __xorshf96_x=(unsigned long) rand();
    __xorshf96_y=(unsigned long) rand();
    __xorshf96_z=(unsigned long) rand();
    unsigned long mask = LNBITSTOMASK(bit_length);
        for (unsigned long i = 0; i < max_size; i++){
            host_data[i] = xorshf96() & mask;
            printf("%l",host_data[i]);
        }
	//COPYPASTA END
    std::cout << "\n";

    cudaMemcpy(dev_data, host_data, data_size, cudaMemcpyHostToDevice);
    cudaMemset(dev_out, 0, compressed_data_size);
    cudaMemset(dev_data_block_start, 0, compression_blocks_count * sizeof(unsigned long));

	/*auto compressed_data = CudaPtr<int>::make_shared(compressed_data_size);
	auto dev_data_block_start = CudaPtr<int>::make_shared(compression_blocks_count);
	auto decompressed_data = CudaPtr<int>::make_shared(max_size);
	dev_data_block_start->reset(compression_blocks_count * sizeof(int));
	decompressed_data->reset(data->size() * sizeof(int));*/

    std::cout << "Compressing...\n";
    run_delta_afl_compress_gpu <int, WARP_SIZE> (bit_length, dev_data, dev_out, dev_data_block_start, max_size);
    std::cout << "Decompressing...\n";
    cudaMemset(dev_data, 0, data_size);
    run_delta_afl_decompress_gpu <int, WARP_SIZE> (bit_length, dev_out, dev_data_block_start, dev_data, max_size);
    cudaMemset(host_data2, 0, data_size);
    cudaMemcpy(host_data2, dev_data, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_data_block_start, dev_data_block_start, compression_blocks_count * sizeof(unsigned long), cudaMemcpyDeviceToHost);

	/*cudaArray.Print(data->copy(100), "encoded");
	cudaArray.Print(dev_data_block_start->copy(compression_blocks_count > 10 ? 10 : compression_blocks_count), "helper");
	cudaArray.Print(decompressed_data->copy(100), "decoded");

    EXPECT_TRUE(CompareDeviceArrays(data->get(), decompressed_data->get(), data->size()));*/
    std::cout << "Decompressed";
    for(int i = 0 ; i <  100; i++){
    	std::cout << host_data2[i];
    }
    std::cout << "\n";

    EXPECT_TRUE(memcmp(host_data, host_data2, max_size *sizeof(int)));

    cudaFree(dev_out);
    cudaFree(dev_data);
    cudaFree(dev_data_block_start);
    cudaFreeHost(host_data);
    cudaFreeHost(host_data2);
    cudaFreeHost(host_data_block_start);
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
