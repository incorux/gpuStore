#include "compression/delta_afl/delta_afl_encoding.hpp"
#include "delta_afl_gpu.cuh"
#include "core/cuda_array.hpp"
#include "util/statistics/cuda_array_statistics.hpp"
#include "util/transform/cuda_array_transform.hpp"
#include "util/copy/cuda_array_copy.hpp"
#include "util/stencil/stencil.hpp"
#include "core/float_cast.hpp"
#include "core/cuda_launcher.cuh"
#include "core/cuda_macros.cuh"

namespace ddj {

__global__ void _delta_afl_splitFloatKernel(float* data, size_t size, int* mantissa,
		int* exponent, int* sign) {
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size)
		return;
	floatCastUnion fu { data[idx] };

	mantissa[idx] = fu.parts.mantisa;
	exponent[idx] = fu.parts.exponent;
	sign[idx] = fu.parts.sign;
}

__global__ void _delta_afl_composeFloatKernel(int* mantissa, int* exponent, int* sign,
		size_t size, float* result) {
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= size)
		return;
	floatCastUnion fu;
	fu.parts.sign = sign[idx];
	fu.parts.exponent = exponent[idx];
	fu.parts.mantisa = mantissa[idx];
	result[idx] = fu.value;
}

template<typename T>
SharedCudaPtrVector<char> DeltaAflEncoding::Encode(SharedCudaPtr<T> data) {
	CUDA_ASSERT_RETURN(cudaGetLastError());
	LOG4CPLUS_INFO_FMT(_logger, "DELTA AFL encoding START: data size = %lu",
			data->size());

	LOG4CPLUS_TRACE_FMT(_logger, "DELTA AFL data to encode: %s",
			CudaArray().ToString(data->copy()).c_str());

	if (data->size() <= 0)
		return SharedCudaPtrVector<char> { CudaPtr<char>::make_shared(),
				CudaPtr<char>::make_shared() };
	// Get minimal bit count needed to encode data
	char minBit = CudaArrayStatistics().MinBitCnt<T>(data) + 1; // (char)13;
	//SharedCudaPtr<int> initial_data = CudaArrayGenerator().GenerateDescendingDeviceArray(max_size);
	const int WARP_SIZE = 32;
	int cword = sizeof(T) * 8;
	unsigned int bit_length = CudaArrayStatistics().MinBitCnt<T>(data);
	int max_size = data->size();
	unsigned long data_size =  max_size * sizeof(int);
	unsigned long data_chunk = cword * WARP_SIZE;
	unsigned long compressed_data_size = (max_size < data_chunk ? data_chunk : max_size);
	compressed_data_size = ((compressed_data_size * bit_length + (data_chunk)-1) / (data_chunk)) * WARP_SIZE * sizeof(T) + (cword) * sizeof(T);
	int compression_blocks_count = (compressed_data_size + (sizeof(T) * WARP_SIZE) - 1) / (sizeof(T) * WARP_SIZE);

	auto result = CudaPtr<char>::make_shared(compressed_data_size);
	auto metadata = CudaPtr<char>::make_shared(4 * sizeof(char));
	auto dataBlockStart= CudaPtr<char>::make_shared(compression_blocks_count * sizeof(T));

	char* host_metadata;
	CUDA_CALL(cudaMallocHost(&host_metadata, 4));
	host_metadata[0] = minBit;

	run_delta_afl_compress_gpu<T, 32>(minBit, data->get(), (T*) result->get(), (T*) dataBlockStart->get(),
			compressed_data_size / sizeof(T));

	metadata->fillFromHost(host_metadata, 4 * sizeof(char));
	CUDA_CALL(cudaFreeHost(host_metadata));

	cudaDeviceSynchronize();
	CUDA_ASSERT_RETURN(cudaGetLastError());
	LOG4CPLUS_INFO(_logger, "DELTA AFL encoding END");

	return SharedCudaPtrVector<char> { metadata, result, dataBlockStart };
}

template<typename T>
SharedCudaPtr<T> DecodeDeltaAfl(T* data, T* dataBlockStart, size_t size, unsigned int minBit) {
	// Calculate length
	long long comprBits = size * 8;
	unsigned long length = comprBits / minBit;

	auto result = CudaPtr<T>::make_shared(length);
	run_delta_afl_decompress_gpu<T, 32>(minBit, data, dataBlockStart, (T*) result->get(), length);
	cudaDeviceSynchronize();
	CUDA_ASSERT_RETURN(cudaGetLastError());

	return result;
}

template<typename T>
SharedCudaPtr<T> DeltaAflEncoding::Decode(SharedCudaPtrVector<char> input) {
	LOG4CPLUS_INFO_FMT(_logger,
			"DELTA AFL decoding START: input[0] size = %lu, input[1] size = %lu",
			input[0]->size(), input[1]->size());

	if (input[1]->size() <= 0)
		return CudaPtr<T>::make_shared();

	auto metadata = input[0]->copyToHost();
	auto data = input[1];
	auto dataBlockStart = input[2];

	// Get min bit and rest
	int minBit = (*metadata)[0];

	// Perform decoding
	auto result = DecodeDeltaAfl<T>((T*) data->get(), (T*) dataBlockStart->get(), data->size(), minBit);

	LOG4CPLUS_INFO(_logger, "DELTA AFL decoding END");
	return result;
}

template<>
SharedCudaPtrVector<char> DeltaAflEncoding::Encode(SharedCudaPtr<float> data) {
	CUDA_ASSERT_RETURN(cudaGetLastError());
	LOG4CPLUS_INFO_FMT(_logger, "DELTA AFL (FLOAT) encoding START: data size = %lu",
			data->size());

	if (data->size() <= 0)
		return SharedCudaPtrVector<char> { CudaPtr<char>::make_shared(),
				CudaPtr<char>::make_shared() };

	auto minMax = CudaArrayStatistics().MinMax(data);
	char allPositive = std::get < 0 > (minMax) >= 0 ? 1 : 0;
	char allNegative = std::get < 1 > (minMax) < 0 ? 2 : 0;
	char sign = allPositive + allNegative;
	auto signResult = CudaPtr<int>::make_shared(data->size());
	auto exponentResult = CudaPtr<int>::make_shared(data->size());
	auto mantissaResult = CudaPtr<int>::make_shared(data->size());

	// Now we split every float number to three integers - sign, exponent and mantissa
	this->_policy.setSize(data->size());
	cudaLaunch(this->_policy, _delta_afl_splitFloatKernel, data->get(), data->size(),
			mantissaResult->get(), exponentResult->get(), signResult->get());
	cudaDeviceSynchronize();

	// We do AFL encoding on mantissa and exponent int arrays
	auto resultVector = Encode(mantissaResult);
	auto resultVector2 = Encode(exponentResult);

	resultVector.insert(resultVector.end(), resultVector2.begin(),
			resultVector2.end());

	// Save the size of mantissa after compression
	// When all numbers are positive or negative we save sign only in metadata as one char
	// Else we save a stencil containing which numbers are negative
	SharedCudaPtr<char> metadata;
	metadata = CudaPtr<char>::make_shared(4 * sizeof(size_t) + 1);
	size_t size = resultVector[1]->size();
	size_t size2 = resultVector[2]->size();
	size_t size3 = resultVector2[1]->size();
	size_t size4 = resultVector2[2]->size();
//	std:: cout << "SIZES1: " << size << "\n";
//	std:: cout << "SIZES2: " << size2 << "\n";
//	std:: cout << "SIZES1: " << size3 << "\n";
//	std:: cout << "SIZES2: " << size4 << "\n";

	CUDA_CALL(cudaMemcpy(metadata->get(), &size, sizeof(size_t), CPY_HTD));
	CUDA_CALL(cudaMemcpy(metadata->get() + sizeof(size_t), &size2, sizeof(size_t), CPY_HTD));
	CUDA_CALL(cudaMemcpy(metadata->get() + 2*sizeof(size_t), &size3, sizeof(size_t), CPY_HTD));
	CUDA_CALL(cudaMemcpy(metadata->get() + 3*sizeof(size_t), &size4, sizeof(size_t), CPY_HTD));
	CUDA_CALL(cudaMemcpy(metadata->get() + 4*sizeof(size_t), &sign, 1, CPY_HTD));
	if (sign == 0) {
		auto stencil = Stencil(signResult).pack();
		metadata = CudaArrayCopy().Concatenate(SharedCudaPtrVector<char> {
				metadata, stencil });
	}

	CUDA_ASSERT_RETURN(cudaGetLastError());
	LOG4CPLUS_INFO(_logger, "DELTA AFL (FLOAT) enoding END");

	return SharedCudaPtrVector<char> { metadata, CudaArrayCopy().Concatenate(
			resultVector) };
}

template<>
SharedCudaPtr<float> DeltaAflEncoding::Decode(SharedCudaPtrVector<char> input) {
	LOG4CPLUS_INFO_FMT(_logger,
			"DELTA AFL (FLOAT) decoding START: input[0] size = %lu, input[1] size = %lu",
			input[0]->size(), input[1]->size());

	if (input[1]->size() <= 0)
		return CudaPtr<float>::make_shared();

	int offset = 0, step = sizeof(char);

	auto metadata = input[0];
	auto data = input[1];

	// read metadata information
	char sign;
	long int compressedMantissaSize, compressedMantissaDataStartSize, compressedExponentSize, compressedExponentDataStartSize;
	CUDA_CALL(cudaMemcpy(&compressedMantissaSize, metadata->get(), sizeof(size_t), CPY_DTH));
	CUDA_CALL(cudaMemcpy(&compressedMantissaDataStartSize, metadata->get() + sizeof(size_t), sizeof(size_t), CPY_DTH));
	CUDA_CALL(cudaMemcpy(&compressedExponentSize, metadata->get()+ 2*sizeof(size_t), sizeof(size_t), CPY_DTH));
	CUDA_CALL(cudaMemcpy(&compressedExponentDataStartSize, metadata->get() + 3*sizeof(size_t), sizeof(size_t), CPY_DTH));
	CUDA_CALL(cudaMemcpy(&sign, metadata->get()+ 4*sizeof(size_t), 1, CPY_DTH));

	std::cout << "Size of mantissa : " << compressedMantissaSize << "\n";
	std::cout << "Size of mantissa start : " << compressedMantissaDataStartSize << "\n";
	std::cout << "Size of exponent : " << compressedExponentSize << "\n";
	std::cout << "Size of exponent start: " << compressedExponentDataStartSize << "\n";

	// read mantissa metadata information
	char minBit, rest;
	CUDA_CALL(cudaMemcpy(&minBit, data->get()+offset, step, CPY_DTH));
	offset += step;
	CUDA_CALL(cudaMemcpy(&rest, data->get()+offset, step, CPY_DTH));
	offset += 3 * step;

	// decode mantissa
	auto mantissaDecoded = DecodeDeltaAfl<int>((int*) (data->get() + offset), (int*) (data->get() + offset + compressedMantissaSize),
			compressedMantissaSize, minBit);
	//long int compressedExponentSize = data->size() - compressedMantissaSize - 8;
	offset += compressedMantissaSize + compressedMantissaDataStartSize;

	// read exponent metadata information
	CUDA_CALL(cudaMemcpy(&minBit, data->get()+offset, step, CPY_DTH));
	offset += step;
	CUDA_CALL(cudaMemcpy(&rest, data->get()+offset, step, CPY_DTH));
	offset += 3 * step;

	// decode exponent
	auto exponentDecoded = DecodeDeltaAfl<int>((int*) (data->get() + offset), (int*) (data->get() + offset + compressedExponentSize ),
			compressedExponentSize, minBit);

	// recover signs
	Stencil stencil;
	size_t size = mantissaDecoded->size();
	if (sign)
		stencil = Stencil(
				CudaArrayTransform().Transform<int, int>(
						CudaPtr<int>::make_shared(size),
						FillOperator<int, int> { (int) sign - 1 }));
	else
		stencil = Stencil(metadata, sizeof(size_t) + 1);

	// compose exponent, mantissa and sign to floats
	auto result = CudaPtr<float>::make_shared(size);
	this->_policy.setSize(size);
	cudaLaunch(this->_policy, _delta_afl_composeFloatKernel, mantissaDecoded->get(),
			exponentDecoded->get(), stencil->get(), size, result->get());
	cudaDeviceSynchronize();
	CUDA_ASSERT_RETURN(cudaGetLastError());
	LOG4CPLUS_INFO(_logger, "DELTA AFL decoding END");

	return result;
}

SharedCudaPtrVector<char> DeltaAflEncoding::EncodeInt(SharedCudaPtr<int> data) {
	return this->Encode<int>(data);
}
SharedCudaPtr<int> DeltaAflEncoding::DecodeInt(SharedCudaPtrVector<char> data) {
	return this->Decode<int>(data);
}
SharedCudaPtrVector<char> DeltaAflEncoding::EncodeTime(SharedCudaPtr<time_t> data) {
	return this->Encode<time_t>(data);
}
SharedCudaPtr<time_t> DeltaAflEncoding::DecodeTime(SharedCudaPtrVector<char> data) {
	return this->Decode<time_t>(data);
}
SharedCudaPtrVector<char> DeltaAflEncoding::EncodeFloat(SharedCudaPtr<float> data) {
	return this->Encode<float>(data);
}
SharedCudaPtr<float> DeltaAflEncoding::DecodeFloat(SharedCudaPtrVector<char> data) {
	return this->Decode<float>(data);
}
SharedCudaPtrVector<char> DeltaAflEncoding::EncodeDouble(
		SharedCudaPtr<double> data) {
	return SharedCudaPtrVector<char>();
}
SharedCudaPtr<double> DeltaAflEncoding::DecodeDouble(
		SharedCudaPtrVector<char> data) {
	return SharedCudaPtr<double>();
}
SharedCudaPtrVector<char> DeltaAflEncoding::EncodeShort(SharedCudaPtr<short> data) {
	return this->Encode<short>(data);
}
SharedCudaPtr<short> DeltaAflEncoding::DecodeShort(SharedCudaPtrVector<char> data) {
	return this->Decode<short>(data);
}
SharedCudaPtrVector<char> DeltaAflEncoding::EncodeChar(SharedCudaPtr<char> data) {
	return this->Encode<char>(data);
}
SharedCudaPtr<char> DeltaAflEncoding::DecodeChar(SharedCudaPtrVector<char> data) {
	return this->Decode<char>(data);
}

size_t DeltaAflEncoding::GetMetadataSize(SharedCudaPtr<char> data, DataType type) {
	if (data->size() <= 0)
		return 0;
	switch (type) {
	case DataType::d_int:
		return 4 * sizeof(char);
	case DataType::d_float:
		return sizeof(size_t) + 1;
	default:
		throw NotImplementedException(
				"No DictEncoding::GetCompressedSize implementation for that type");
	}
}

size_t DeltaAflEncoding::GetCompressedSize(SharedCudaPtr<char> data, DataType type) {
	if (data->size() <= 0)
		return 0;
	switch (type) {
	case DataType::d_int:
		return GetCompressedSizeIntegral(CastSharedCudaPtr<char, int>(data));
	case DataType::d_float:
		return GetCompressedSizeFloatingPoint(
				CastSharedCudaPtr<char, float>(data));
	default:
		throw NotImplementedException(
				"No DictEncoding::GetCompressedSize implementation for that type");
	}
}

template<typename T>
size_t DeltaAflEncoding::GetCompressedSizeIntegral(SharedCudaPtr<T> data) {
	char minBit = CudaArrayStatistics().MinBitCnt<T>(data) + 1; // (char)13;
	//SharedCudaPtr<int> initial_data = CudaArrayGenerator().GenerateDescendingDeviceArray(max_size);
	const int WARP_SIZE = 32;
	int cword = sizeof(T) * 8;
	unsigned int bit_length = CudaArrayStatistics().MinBitCnt<T>(data);
	int max_size = data->size();
	unsigned long data_size =  max_size * sizeof(int);
	unsigned long data_chunk = cword * WARP_SIZE;
	unsigned long compressed_data_size = (max_size < data_chunk ? data_chunk : max_size);
	compressed_data_size = ((compressed_data_size * bit_length + (data_chunk)-1) / (data_chunk)) * WARP_SIZE * sizeof(T) + (cword) * sizeof(T);
	int compression_blocks_count = (compressed_data_size + (sizeof(T) * WARP_SIZE) - 1) / (sizeof(T) * WARP_SIZE);
	return compressed_data_size + compression_blocks_count;
}

template<typename T>
size_t DeltaAflEncoding::GetCompressedSizeFloatingPoint(SharedCudaPtr<T> data) {
	auto minMax = CudaArrayStatistics().MinMax(data);
	auto signResult = CudaPtr<int>::make_shared(data->size());
	auto exponentResult = CudaPtr<int>::make_shared(data->size());
	auto mantissaResult = CudaPtr<int>::make_shared(data->size());

	// Now we split every float number to three integers - sign, exponent and mantissa
	this->_policy.setSize(data->size());
	cudaLaunch(this->_policy, _delta_afl_splitFloatKernel, data->get(), data->size(),
			mantissaResult->get(), exponentResult->get(), signResult->get());
	cudaDeviceSynchronize();

	size_t size = GetCompressedSizeIntegral(exponentResult)
			+ GetCompressedSizeIntegral(mantissaResult);
	size += GetMetadataSize(CastSharedCudaPtr<int, char>(exponentResult),
			DataType::d_int);
	size += GetMetadataSize(CastSharedCudaPtr<int, char>(mantissaResult),
			DataType::d_int);
	return size;
}

#define DELTA_AFL_ENCODING_SPEC(X) \
	template SharedCudaPtrVector<char> DeltaAflEncoding::Encode<X>(SharedCudaPtr<X>); \
	template SharedCudaPtr<X> DeltaAflEncoding::Decode<X>(SharedCudaPtrVector<char>);
FOR_EACH(DELTA_AFL_ENCODING_SPEC, char, short, int, long, unsigned int)

} /* namespace ddj */
