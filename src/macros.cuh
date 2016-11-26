#ifndef macros
#define macros 0

// This should work independently from _CUDA_ARCH__ number
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

//TODO: distinguish between signed/unsigned versions

// This depend on _CUDA_ARCH__ number



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

__device__ __host__ __forceinline__ unsigned int BITLEN(unsigned int word) 
{ 
    unsigned int ret=0; 
#if __CUDA_ARCH__ > 200 
    asm volatile ("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
#else
    while (word >>= 1) 
      ret++;
#endif
   return ret > 64 ? 0 : ret;
}

__device__ __host__ __forceinline__ unsigned int BITLEN(unsigned long word) 
{ 
    unsigned int ret=0; 
#if __CUDA_ARCH__ > 200 
    asm volatile ("bfind.u64 %0, %1;" : "=r"(ret) : "l"(word));
#else
    while (word >>= 1) 
      ret++;
#endif
   return ret > 64 ? 0 : ret;
}

__device__ __host__ __forceinline__ unsigned int BITLEN(int word) 
{ 
    unsigned int ret=0; 
#if __CUDA_ARCH__ > 200 
    asm volatile ("bfind.s32 %0, %1;" : "=r"(ret) : "r"(word));
#else
    while (word >>= 1) 
      ret++;
#endif
   return ret > 64 ? 0 : ret;
}

__device__ __host__ __forceinline__ unsigned int BITLEN(long word) 
{ 
    unsigned int ret=0; 
#if __CUDA_ARCH__ > 200 
    asm volatile ("bfind.s64 %0, %1;" : "=r"(ret) : "l"(word));
#else
    while (word >>= 1) 
      ret++;
#endif
   return ret > 64 ? 0 : ret;
}

__host__ __device__
inline int ALT_BITLEN( int v)
{
    register unsigned int r; // result of log2(v) will go here
    register unsigned int shift;

    r =     (v > 0xFFFF) << 4; v >>= r;
    shift = (v > 0xFF  ) << 3; v >>= shift; r |= shift;
    shift = (v > 0xF   ) << 2; v >>= shift; r |= shift;
    shift = (v > 0x3   ) << 1; v >>= shift; r |= shift;
    r |= (v >> 1);
    return r+1;
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

#define SGN(a) (int)((unsigned int)((int)a) >> (sizeof(int) * CHAR_BIT - 1))
#define GETNSGNBITS(a,n,b) ((SGN(a) << (n-1)) | GETNBITS(((a)>>(b-n)), (n-1))) 

// Make a FOREACH macro
#define FE_1(WHAT, X) WHAT(X) 
#define FE_2(WHAT, X, ...) WHAT(X)FE_1(WHAT, __VA_ARGS__)
#define FE_3(WHAT, X, ...) WHAT(X)FE_2(WHAT, __VA_ARGS__)
#define FE_4(WHAT, X, ...) WHAT(X)FE_3(WHAT, __VA_ARGS__)
#define FE_5(WHAT, X, ...) WHAT(X)FE_4(WHAT, __VA_ARGS__)
#define FE_6(WHAT, X, ...) WHAT(X)FE_5(WHAT, __VA_ARGS__)
//... repeat as needed

#define GET_MACRO(_1,_2,_3,_4,_5,NAME,...) NAME 
#define FOR_EACH(action,...) \
  GET_MACRO(__VA_ARGS__,FE_5,FE_4,FE_3,FE_2,FE_1)(action,__VA_ARGS__)

#endif
