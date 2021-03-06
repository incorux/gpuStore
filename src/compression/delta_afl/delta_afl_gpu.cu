#include "delta_afl_gpu.cuh"
#include "core/cuda_macros.cuh"
#include "core/macros.h"
#include <stdio.h>

template <typename T, char CWARP_SIZE>
__device__  void delta_afl_compress_base_gpu(
        const unsigned int bit_length,
        unsigned long data_id,
        unsigned long comp_data_id,
        T *data,
        T *compressed_data,
        T* compressed_data_block_start,
        unsigned long length
        )
{
    if (data_id >= length) return;

    // TODO: Compressed data should be always unsigned, fix that latter
    T v1;
    unsigned int uv1;
    unsigned int value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;
    unsigned int sgn = 0;

    T zeroLaneValue, v2;
    const unsigned long lane = get_lane_id();
    char neighborId = lane - 1;

    const unsigned long data_block = ( blockIdx.x * blockDim.x) / CWARP_SIZE + threadIdx.x / CWARP_SIZE;

    if (lane == 0 )  {
        neighborId = 31;
        zeroLaneValue = data[pos_data];
        compressed_data_block_start[data_block] = zeroLaneValue;
    }

    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_data < length; ++i)
    {
        v1 = data[pos_data];
        pos_data += CWARP_SIZE;

        v2 = shfl_get_value(v1, neighborId);

        if (lane == 0)
        {
            // Lane 0 uses data from previous iteration
            v1 = zeroLaneValue - v1;
            zeroLaneValue = v2;
        } else {
            v1 = v2 - v1;
        }

        //TODO: ugly hack, fix that with correct bfe calls
        sgn = ((unsigned int) v1) >> 31;
        uv1 = max(v1, -v1);
        // END: ugly hack

        if (v1_pos >= CWORD_SIZE(T) - bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;
            value = value | (GETNBITS(uv1, v1_len) << v1_pos);

            if (v1_pos == CWORD_SIZE(T) - bit_length) // whole word
                value |= (GETNBITS(uv1, v1_len - 1) | (sgn << (v1_len - 1))) << (v1_pos);
            else // begining of the word
                value |= GETNBITS(uv1, v1_len) << (v1_pos);

            compressed_data[pos] = reinterpret_cast<int&>(value);

            v1_pos = bit_length - v1_len;
            value = 0;
            // if is necessary as otherwise may work with negative bit shifts
            if (v1_pos > 0) // The last part of the word
                value = (GETNPBITS(uv1, v1_pos - 1, v1_len)) | (sgn << (v1_pos - 1));

            pos += CWARP_SIZE;
        } else {
            v1_len = bit_length;
            value |= (GETNBITS(uv1, v1_len-1) | (sgn << (v1_len-1))) << v1_pos;
            v1_pos += v1_len;
        }
    }

    if (pos_data >= length  && pos_data < length + CWARP_SIZE)
    {
        compressed_data[pos] = reinterpret_cast<int&>(value);
    }
}

template <typename T, char CWARP_SIZE>
__device__ void delta_afl_decompress_base_gpu(
        const unsigned int bit_length,
        unsigned long comp_data_id,
        unsigned long data_id,
        T *compressed_data,
        T* compressed_data_block_start,
        T *data,
        unsigned long length
        )
{
    unsigned long pos = comp_data_id, pos_decomp = data_id;
    unsigned int v1_pos = 0, v1_len;
    unsigned int v1;
    unsigned int ret;
    int sret;

    const unsigned long lane = get_lane_id();

    if (pos_decomp >= length ) // Decompress not more elements then length
        return;

    v1 = reinterpret_cast<unsigned int &>(compressed_data[pos]);

    T zeroLaneValue = 0, v2 = 0;

    const unsigned long data_block = (blockIdx.x * blockDim.x) / CWARP_SIZE  + threadIdx.x / CWARP_SIZE;

    if (lane == 0) {
       zeroLaneValue = compressed_data_block_start[data_block];
    }

    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_decomp < length; ++i)
    {
        if (v1_pos >= CWORD_SIZE(T) - bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;
            ret = GETNPBITS(v1, v1_len, v1_pos);

            pos += CWARP_SIZE;
            v1 = reinterpret_cast<unsigned int &>(compressed_data[pos]);

            v1_pos = bit_length - v1_len;
            ret = ret | ((GETNBITS(v1, v1_pos))<< v1_len);
        } else {
            v1_len = bit_length;
            ret = GETNPBITS(v1, v1_len, v1_pos);
            v1_pos += v1_len;
        }

        // TODO: dirty hack
        int sgn_multiply = (ret >> (bit_length-1)) ? -1 : 1;
        // END
        ret &= NBITSTOMASK(bit_length-1);
        sret = sgn_multiply * (int)(ret);

        sret = shfl_prefix_sum(sret); // prefix sum deltas

        v2 = shfl_get_value(zeroLaneValue, 0);
        sret = v2 - sret;

        data[pos_decomp] = sret;
        pos_decomp += CWARP_SIZE;

        v2 = shfl_get_value(sret, 31);

        if(lane == 0)
            zeroLaneValue = v2;
    }
}

template < typename T, char CWARP_SIZE >
__global__ void delta_afl_compress_gpu (const unsigned int bit_length, T *data, T *compressed_data, T* compressed_data_block_start, unsigned long length)
{
    const unsigned int warp_lane = get_lane_id();
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
    const unsigned long cdata_id = data_block * bit_length + warp_lane;

    delta_afl_compress_base_gpu <T, CWARP_SIZE> (bit_length, data_id, cdata_id, data, compressed_data, compressed_data_block_start, length);
}

template < typename T, char CWARP_SIZE >
__global__ void delta_afl_decompress_gpu (const unsigned int bit_length, T *compressed_data, T* compressed_data_block_start, T * decompress_data, unsigned long length)
{
    const unsigned int warp_lane = get_lane_id();
    const unsigned long data_block = blockIdx.x * blockDim.x + threadIdx.x - warp_lane;
    const unsigned long data_id = data_block * CWORD_SIZE(T) + warp_lane;
    const unsigned long cdata_id = data_block * bit_length + warp_lane;

    delta_afl_decompress_base_gpu <T, CWARP_SIZE> (bit_length, cdata_id, data_id, compressed_data, compressed_data_block_start, decompress_data, length);
}

template < typename T, char CWARP_SIZE >
__host__ void run_delta_afl_compress_gpu(const unsigned int bit_length, T *data, T *compressed_data, T* compressed_data_block_start, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy 
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    delta_afl_compress_gpu <T, CWARP_SIZE> <<<block_number, block_size>>> (bit_length, data, compressed_data, compressed_data_block_start,length);
}

template < typename T, char CWARP_SIZE >
__host__ void run_delta_afl_decompress_gpu(const unsigned int bit_length, T *compressed_data, T* compressed_data_block_start, T *data, unsigned long length)
{
    const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
    const unsigned long block_number = (length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));
    delta_afl_decompress_gpu <T, CWARP_SIZE> <<<block_number, block_size>>> (bit_length, compressed_data, compressed_data_block_start,data, length);
}

#define DELTA_GFL_SPEC(X, A) \
template  __host__  void run_delta_afl_decompress_gpu<X, A> (const unsigned int bit_length, X *compressed_data, X* compressed_data_block_start, X *data, unsigned long length);\
template  __host__  void run_delta_afl_compress_gpu<X, A> (const unsigned int bit_length, X *data, X *compressed_data, X* compressed_data_block_start, unsigned long length);\
template  __global__  void delta_afl_decompress_gpu <X, A> (const unsigned int bit_length, X *compressed_data, X* compressed_data_block_start, X * decompress_data, unsigned long length);\
template  __global__  void delta_afl_compress_gpu <X, A> (const unsigned int bit_length, X *data, X *compressed_data, X* compressed_data_block_start, unsigned long length);\
template __device__  void delta_afl_decompress_base_gpu <X, A> ( const unsigned int bit_length, unsigned long comp_data_id, unsigned long data_id, X *compressed_data, X* compressed_data_block_start, X *data, unsigned long length);\
template __device__   void delta_afl_compress_base_gpu <X, A> (const unsigned int bit_length, unsigned long data_id, unsigned long comp_data_id, X *data, X *compressed_data, X* compressed_data_block_start, unsigned long length);

#define DELTA_AFL_SPEC(X) DELTA_GFL_SPEC(X, 32)
FOR_EACH(DELTA_AFL_SPEC, char, short, int, long, unsigned int)

#define DELTA_FL_SPEC(X) DELTA_GFL_SPEC(X, 1)
FOR_EACH(DELTA_FL_SPEC, char, short, int, long, unsigned int)
