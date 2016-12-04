#ifndef DDJ_DELTA_AFL_ENCODING_HPP_
#define DDJ_DELTA_AFL_ENCODING_HPP_

#include "compression/encoding.hpp"
#include "compression/encoding_factory.hpp"
#include "core/execution_policy.hpp"

namespace ddj
{

class DeltaAflEncoding : public Encoding
{
public:
	DeltaAflEncoding() : Encoding("Encoding.DeltaAfl") {}
	~DeltaAflEncoding(){}
	DeltaAflEncoding(const DeltaAflEncoding&) = default;
	DeltaAflEncoding(DeltaAflEncoding&&) = default;

public:
	unsigned int GetNumberOfResults() { return 1; }

	size_t GetMetadataSize(SharedCudaPtr<char> data, DataType type);
	size_t GetCompressedSize(SharedCudaPtr<char> data, DataType type);

protected:
	SharedCudaPtrVector<char> EncodeInt(SharedCudaPtr<int> data);
	SharedCudaPtr<int> DecodeInt(SharedCudaPtrVector<char> data);
	SharedCudaPtrVector<char> EncodeTime(SharedCudaPtr<time_t> data);
	SharedCudaPtr<time_t> DecodeTime(SharedCudaPtrVector<char> data);
	SharedCudaPtrVector<char> EncodeFloat(SharedCudaPtr<float> data);
	SharedCudaPtr<float> DecodeFloat(SharedCudaPtrVector<char> data);
	SharedCudaPtrVector<char> EncodeDouble(SharedCudaPtr<double> data);
	SharedCudaPtr<double> DecodeDouble(SharedCudaPtrVector<char> data);
	SharedCudaPtrVector<char> EncodeShort(SharedCudaPtr<short> data);
	SharedCudaPtr<short> DecodeShort(SharedCudaPtrVector<char> data);
	SharedCudaPtrVector<char> EncodeChar(SharedCudaPtr<char> data);
	SharedCudaPtr<char> DecodeChar(SharedCudaPtrVector<char> data);

	template<typename T>
	size_t GetCompressedSizeIntegral(SharedCudaPtr<T> data);

	template<typename T>
	size_t GetCompressedSizeFloatingPoint(SharedCudaPtr<T> data);

public:
	template<typename T> SharedCudaPtrVector<char> Encode(SharedCudaPtr<T> data);
	template<typename T> SharedCudaPtr<T> Decode(SharedCudaPtrVector<char> data);

private:
	ExecutionPolicy _policy;
};

class DeltaAflEncodingFactory : public EncodingFactory
{
public:
	DeltaAflEncodingFactory(DataType dt)
		: EncodingFactory(dt, EncodingType::deltaafl)
	{}
	~DeltaAflEncodingFactory(){}
	DeltaAflEncodingFactory(const DeltaAflEncodingFactory& other)
		: EncodingFactory(other.dataType, EncodingType::deltaafl)
	{}

	boost::shared_ptr<Encoding> Get()
	{
		return boost::make_shared<DeltaAflEncoding>();
	}

	boost::shared_ptr<Encoding> Get(SharedCudaPtr<char> data)
	{
		return Get();
	}
};

} /* namespace ddj */
#endif /* DDJ_DELTA_AFL_ENCODING_HPP_ */
