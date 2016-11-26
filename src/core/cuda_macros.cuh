#ifndef DDJ_CUDA_MACROS_CUH_
#define DDJ_CUDA_MACROS_CUH_

#define CWORD_SIZE(T)(T) (sizeof(T) * 8)

#define NBITSTOMASK(n) ((1<<(n)) - 1)
#define LNBITSTOMASK(n) ((1L<<(n)) - 1)

#define fillto(b,c) (((c + b - 1) / b) * b)

#define _unused(x) x __attribute__((unused))
#define convert_struct(n, s)  struct sgn {signed int x:n;} __attribute__((unused)) s

__device__ inline int get_lane_id(int warp_size=32) { return threadIdx.x % warp_size; } //TODO: move to macros and reuse

__inline__ __device__
int warpAllReduceMax(int val) {

    val = max(val, __shfl_xor(val,16));
    val = max(val, __shfl_xor(val, 8));
    val = max(val, __shfl_xor(val, 4));
    val = max(val, __shfl_xor(val, 2));
    val = max(val, __shfl_xor(val, 1));

    /*int m = val;*/
    /*for (int mask = warpSize/2; mask > 0; mask /= 2) {*/
        /*m = __shfl_xor(val, mask);*/
        /*val = m > val ? m : val;*/
    /*}*/
    return val;
}

/*
 *
 *				SETNPBITS
 *
 */

template <typename T>
__device__ __host__ __forceinline__ T SETNPBITS( T *source, T value, const unsigned int num_bits, const unsigned int bit_start)
{
    T mask = NBITSTOMASK(num_bits);
    *source &= ~(mask<<bit_start); // clear space in source
    *source |= (value & mask) << bit_start; // set values
    return *source;
}

__device__ __host__ __forceinline__ long SETNPBITS( long *source, long value, unsigned int num_bits, unsigned int bit_start)
{
    long mask = LNBITSTOMASK(num_bits);
    *source &= ~(mask<<bit_start); // clear space in source
    *source |= (value & mask) << bit_start; // set values
    return *source;
}

__device__ __host__ __forceinline__ unsigned long SETNPBITS( unsigned long *source, unsigned long value, unsigned int num_bits, unsigned int bit_start)
{
    unsigned long mask = LNBITSTOMASK(num_bits);
    *source &= ~(mask<<bit_start); // clear space in source
    *source |= (value & mask) << bit_start; // set values
    return *source;
}

/*
 *
 *				GETNPBITS
 *
 */

__device__ __host__ __forceinline__ unsigned int GETNPBITS( int source, unsigned int num_bits, unsigned int bit_start)
{
#if __CUDA_ARCH__ > 200  // This improves performance
    unsigned int bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(bit_start), "r"(num_bits));
    return bits;
#else
    return ((source>>bit_start) & NBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned int GETNPBITS( unsigned int source, unsigned int num_bits, unsigned int bit_start)
{
#if __CUDA_ARCH__ > 200  // This improves performance
    unsigned int bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"((unsigned int) source), "r"(bit_start), "r"(num_bits));
    return bits;
#else
    return ((source>>bit_start) & NBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned long GETNPBITS( long source, unsigned int num_bits, unsigned int bit_start)
{
#if __CUDA_ARCH__ > 200  // This improves performance
    unsigned long bits;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(bits) : "l"((unsigned long) source), "r"(bit_start), "r"(num_bits));
    return bits;
#else
    return ((source>>bit_start) & LNBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned long GETNPBITS( unsigned long source, unsigned int num_bits, unsigned int bit_start)
{
#if __CUDA_ARCH__ > 200  // This improves performance
    unsigned long bits;
    asm("bfe.u64 %0, %1, %2, %3;" : "=l"(bits) : "l"((unsigned long) source), "r"(bit_start), "r"(num_bits));
    return bits;
#else
    return ((source>>bit_start) & LNBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned char GETNPBITS( char source, unsigned int num_bits, unsigned int bit_start)
{ return ((source>>bit_start) & NBITSTOMASK(num_bits)); }

__device__ __host__ __forceinline__ unsigned short GETNPBITS( short source, unsigned int num_bits, unsigned int bit_start)
{ return ((source>>bit_start) & NBITSTOMASK(num_bits)); }

/*
 *
 *				GETNBITS
 *
 */

__device__ __host__ __forceinline__ unsigned long GETNBITS( long source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else // In other case this will be faster
    return ((source) & LNBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned long GETNBITS( unsigned long source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else // In other case this will be faster
    return ((source) & LNBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned int GETNBITS( int source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else // In other case this will be faster
    return ((source) & NBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned int GETNBITS( unsigned int source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else // In other case this will be faster
    return ((source) & NBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned char GETNBITS( char source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else // In other case this will be faster
    return ((source) & NBITSTOMASK(num_bits));
#endif
}

__device__ __host__ __forceinline__ unsigned short GETNBITS( short source, unsigned int num_bits)
{
#if __CUDA_ARCH__ > 200  // Use bfe implementation
    return GETNPBITS(source, num_bits, 0);
#else // In other case this will be faster
    return ((source) & NBITSTOMASK(num_bits));
#endif
}

/*
 *
 *				BITLEN
 *
 */


template<typename T>
inline char BITLEN(T min, T max)
{
	if(min<0) return 8*sizeof(T);
	char ret=0;
	while(max>>=1) ret++;
	return ret+1;
}

template<>
inline char BITLEN(unsigned int min, unsigned int max)
{
	char ret=0;
	while(max>>=1) ret++;
	return ret+1;
}

template<> inline char BITLEN(float min, float max)
{ return 32; }

template<> inline char BITLEN(double min, double max)
{ return 64; }

__host__ __device__
inline int ALT_BITLEN(int v)
{
	if (v < 0) return 8 * sizeof(int); // for negative

    unsigned int r; // result of log2(v) will go here
    unsigned int shift;

    r =     (v > 0xFFFF) << 4; v >>= r;
    shift = (v > 0xFF  ) << 3; v >>= shift; r |= shift;
    shift = (v > 0xF   ) << 2; v >>= shift; r |= shift;
    shift = (v > 0x3   ) << 1; v >>= shift; r |= shift;
    r |= (v >> 1);
    return r+1;
}

/*
 *
 *				OTHERS
 *
 */

#define SGN(a) (int)((unsigned int)((int)a) >> (sizeof(int) * CHAR_BIT - 1))
#define GETNSGNBITS(a,n,b) ((SGN(a) << (n-1)) | GETNBITS(((a)>>(b-n)), (n-1)))

__device__ __host__ __forceinline__
unsigned int SaveNbitIntValToWord(int nbit, int position, int value, unsigned int word)
{
    return word | (value << (nbit * position));
}

__device__ __host__ __forceinline__
int ReadNbitIntValFromWord(int nbit, int position, unsigned int word)
{
    return GETNPBITS(word, nbit, position * nbit);
}

__device__ __host__ __forceinline__
char SetNthBit(int n, int bit, char value)
{
	return value ^ ((-bit ^ value) & (1 << n));
}

__device__ __host__ __forceinline__
bool GetNthBit(int n, char value)
{
	return (value >> n) & 1;
}

__device__ inline long shfl_up(long value, int i, int width=32)
{
    int lo, hi;
    asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"l"(value));

    lo =  __shfl_up(lo, i, width); // add zeroLaneValue
    hi =  __shfl_up(hi, i, width); // add zeroLaneValue

    asm volatile("mov.b64 %0,{%1,%2};":"=l"(value):"r"(lo),"r"(hi));

    return value;
}
__device__ inline int shfl_prefix_sum(int value, int width=32)  // TODO: move to macros and reuse
{
    int lane_id = get_lane_id();

    // Now accumulate in log2(32) steps
#pragma unroll
    for(int i=1; i<=width; i*=2) {
        int n = __shfl_up(value, i);
        if(lane_id >= i) value += n;
    }

    return value;
}


__device__ inline long shfl_prefix_sum(long value, int width=32)  // TODO: move to macros and reuse
{
    int lane_id = get_lane_id();

    // Now accumulate in log2(32) steps
#pragma unroll
    for(int i=1; i<=width; i*=2) {
        long n = shfl_up(value, i);
        if(lane_id >= i) value += n;
    }

    return value;
}

__device__ inline long shfl_prefix_sum(unsigned int value, int width=32)  // TODO: move to macros and reuse
{
    int lane_id = get_lane_id();

    // Now accumulate in log2(32) steps
#pragma unroll
    for(int i=1; i<=width; i*=2) {
        long n = shfl_up(value, i);
        if(lane_id >= i) value += n;
    }

    return value;
}


__device__ inline int shfl_get_value(int value, int laneId, int width=32)
{
    return __shfl(value, laneId, width); // add zeroLaneValue
}

__device__ inline long shfl_get_value(long value, int laneId, int width=32)
{
    int lo, hi;
    asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"l"(value));

    lo =  __shfl(lo, laneId, width); // add zeroLaneValue
    hi =  __shfl(hi, laneId, width); // add zeroLaneValue

    asm volatile("mov.b64 %0,{%1,%2};":"=l"(value):"r"(lo),"r"(hi));

    return value;
}

__device__ inline int shfl_get_value(unsigned int value, int laneId, int width=32)
{
    return __shfl(value, laneId, width); // add zeroLaneValue
}

#define SGN(a) (int)((unsigned int)((int)a) >> (sizeof(int) * CHAR_BIT - 1))
#define GETNSGNBITS(a,n,b) ((SGN(a) << (n-1)) | GETNBITS(((a)>>(b-n)), (n-1)))

#endif /* DDJ_CUDA_MACROS_CUH_ */
