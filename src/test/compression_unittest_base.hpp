/*
 *  compression_unittest_base.hpp
 *
 *  Created on: 22-04-2015
 *      Author: Karol Dzitkowski
 */

#ifndef DDJ_COMPRESSION_UNITTEST_BASE_HPP_
#define DDJ_COMPRESSION_UNITTEST_BASE_HPP_

#include "unittest_base.hpp"
#include "compression/encoding.hpp"

namespace ddj
{

class CompressionUnittestBase : public UnittestBase, public ::testing::WithParamInterface<int>
{
public:
	template<typename T>
	static bool TestSize(
			boost::function<SharedCudaPtrVector<char> (SharedCudaPtr<T> data)> encodeFunction,
			boost::function<SharedCudaPtr<T> (SharedCudaPtrVector<char> data)> decodeFunction,
			SharedCudaPtr<T> data);

	template<typename T>
	static bool TestContent(
			boost::function<SharedCudaPtrVector<char> (SharedCudaPtr<T>)> encodeFunction,
			boost::function<SharedCudaPtr<T> (SharedCudaPtrVector<char>)> decodeFunction,
			SharedCudaPtr<T> data);

	template<class EncodingClass, typename T>
	static void TestGetMetadataSize(SharedCudaPtr<T> data)
	{
		boost::shared_ptr<Encoding> encoder = boost::make_shared<EncodingClass>();
		auto testData = MoveSharedCudaPtr<T, char>(data);
		auto encoded = encoder->Encode(testData, GetDataType<T>());
		auto expected = encoded[0]->size();
		auto actual = encoder->GetMetadataSize(testData, GetDataType<T>());
		EXPECT_EQ(expected, actual);
	}

	template<class EncodingClass, typename T>
	static void TestGetCompressedSize(SharedCudaPtr<T> data)
	{
		boost::shared_ptr<Encoding> encoder = boost::make_shared<EncodingClass>();
		auto testData = MoveSharedCudaPtr<T, char>(data);
		auto encoded = encoder->Encode(testData, GetDataType<T>());
		size_t expected = 0;
		for(int i=1; i <= encoder->GetNumberOfResults(); i++)
			expected += encoded[i]->size();
		auto actual = encoder->GetCompressedSize(testData, GetDataType<T>());
		EXPECT_EQ(expected, actual);
	}
    unsigned long __xorshf96_x=123456789, __xorshf96_y=362436069, __xorshf96_z=521288629;
	inline unsigned long xorshf96(void) {          //period 2^96-1
	// This is only for test purposes so it is optimized for speed (true randomness is not needed)
	    unsigned long t;
	        __xorshf96_x ^= __xorshf96_x << 16;
	        __xorshf96_x ^= __xorshf96_x >> 5;
	        __xorshf96_x ^= __xorshf96_x << 1;

	        t = __xorshf96_x;
	        __xorshf96_x = __xorshf96_y;
	        __xorshf96_y = __xorshf96_z;
	        __xorshf96_z = t ^ __xorshf96_x ^ __xorshf96_y;

	        return __xorshf96_z;
	}
protected:
	void SetUp()
	{
		_size = GetParam();
	}
};

} /* namespace ddj */
#endif /* DDJ_COMPRESSION_UNITTEST_HPP_ */
