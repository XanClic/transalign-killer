#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>

#define DEBUG 1
#define NGRID 512
#define NBLOCK 65535
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}


/* The exponent given here determines the steps taken in the adding kernel. An
 * exponent of 1 results in rounding the size to 2^1 = 2, therefore, in every
 * step, two input fields are added and the size shrinks to half of what it was
 * before. This influences the size of the result buffer as well (the greater
 * this exponent is, the smaller the result will be). */
#define BASE_EXP 4
#define BASE (1 << BASE_EXP)

/* Define this to actually use host memory instead of copying the buffer to the
 * GPU (as it turns out, this may actually be worth it) */
#define USE_HOST_PTR


#ifdef USE_HOST_PTR
#define HOST_PTR_POLICY CL_MEM_USE_HOST_PTR
#else
#define HOST_PTR_POLICY CL_MEM_COPY_HOST_PTR
#endif

/**
 * These two functions provide std::chrono functionality (see cpp-stuff.cpp for
 * an explanation why they're extern).
 CUDA nvcc don't support it.
extern void clock_start(void);
extern long clock_delta(void);
*/

__global__ void k_iadd(unsigned *dest, char *sequence, unsigned seq_length)
{
    for (unsigned id =  blockIdx.x*blockDim.x+threadIdx.x; 
        id < seq_length; 
        id += blockDim.x*gridDim.x)
    {
    
        unsigned result = 0;
        unsigned in_start = id << BASE_EXP;

        if (in_start < seq_length)
        {
            for (unsigned i = in_start; i < in_start + BASE; i++)
            {
                char nucleobase = sequence[i];
                result += nucleobase != '-';
            }
        }

        dest[id] = result;
    }
}

__global__ void k_cadd(unsigned *buffer, unsigned doff, unsigned soff)
{
    unsigned id = blockIdx.x*blockDim.x+threadIdx.x;
    unsigned in_start = soff + (id << BASE_EXP);
    unsigned out_pos = doff + id;
    unsigned result = 0;
    
    for (unsigned i = in_start; i < in_start + BASE; i++)
    {
        unsigned value = buffer[i];
        result += value;
    }
    
    buffer[out_pos] = result;
}

/**
 * Rounds a value x up to the next power of 2^exp.
 */
static long round_up_to_power_of_two(long x, int exp)
{
    assert(x > 0);

    x--;

    int i;
    for (i = 0; x; i++)
        x >>= exp;
    for (x = 1; i; i--)
        x <<= exp;

    return x;
}


/**
 * Loads a text file and returns a buffer with the contents.
 */
static char *load_text(const char *filename, long *length_ptr)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        fprintf(stderr, "Could not load file \"%s\": %s\n", filename, strerror(errno));
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long length = ftell(fp);
    rewind(fp);

    long mem_len = length + 1;

    if (length_ptr)
        *length_ptr = mem_len;

    char *content = (char *)calloc(mem_len, 1);
    fread(content, 1, length, fp);
    fclose(fp);

    return content;
}


int main(int argc, char *argv[])
{
    dim3 grid1d(NGRID,1,1); 
    dim3 block1d(NBLOCK,1,1);
    unsigned clstsi, clstc, clsto, local_index;
    long delta_time;
    struct timeval start_time, end_time;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: transalign_killer [--cldev=x.y] <input file>\n");
        fprintf(stderr, "  --cldev=x.y: x specifies the platform index, y the device index.\n");
        return 1;
    }


    long seq_length;
    //CUDA kernel input
    char *sequence = load_text(argv[argc - 1], &seq_length);

    if (!sequence)
        return 1;
//DEBUG
    for (int i=0; i<seq_length; ++i)
    {
        printf("%c", sequence[i]);
    }
    printf("\n");


    seq_length--; // Cut final 0 byte

    // FIXME: All the following code relies on seq_length being a multiple of BASE.

    long round_seq_length = round_up_to_power_of_two(seq_length, BASE_EXP);

    long res_length = 0;
    for (long len = round_seq_length / BASE; len; len /= BASE)
        res_length += len;

    printf("res_length: %d\n", res_length);

    // Use some random index to be searched for here
    unsigned letter_index = seq_length / 2;

    // Create the result buffer
    // CUDA kernel output
    unsigned *result = (unsigned *)malloc(res_length * sizeof(unsigned));
    unsigned *result_gpu;
    char *seq_gpu;

    //replace clock_start(); with gettimeofday()
    gettimeofday(&start_time, NULL);

#if DEBUG
    printf("GPU part started\n");
#endif
    /*** START OF ROCKET SCIENCE LEVEL RUNTIME-TIME INTENSIVE STUFF ***/

    // Bandwidth intensive stuff goes here
    // Copy the sequence to the video memory (or, generally speaking, the OpenCL device)
    CUDA_CHECK(cudaMalloc((void**)&result_gpu, res_length * sizeof(unsigned)));//result_gpu
    CUDA_CHECK(cudaMalloc((void**)&seq_gpu, seq_length*sizeof(char)));//seq_gpu
    CUDA_CHECK(cudaMemcpy(seq_gpu, sequence, res_length * sizeof(char), cudaMemcpyHostToDevice));

#if DEBUG
    printf("GPU malloc and cpy finised\n");
#endif

    //replace clock_delta(); with gettimeofday()
    gettimeofday(&end_time, NULL);
    long bw1_time = (end_time.tv_sec*1000000+end_time.tv_usec) - (start_time.tv_sec*1000000+start_time.tv_usec);

    // GPU intensive stuff goes here

    /**
     * First, transform every - and \0 into a 0 and every other character into a
     * 1. Then, add consecutive fields (BASE fields) together and store them at
     * the beginning of the result buffer.
     */
    
    //TODO: ADD correct kernel parameters
#if DEBUG
    printf("k_iadd launching\n");
#endif

    k_iadd<<<grid1d,block1d>>>(result_gpu, seq_gpu, seq_length);

#if DEBUG
    printf("k_iadd finished\n");
#endif

    CUDA_CHECK(cudaMemcpy(result, result_gpu, res_length * sizeof(unsigned), cudaMemcpyDeviceToHost));
#if DEBUG
    printf("result back\n");
    for (int i = 0; i < res_length; i++)
    {
        printf("%d ", result[i]);
    }
    printf("\n");
#endif

#if DEBUG
    printf("k_iadd result back\n");
#endif
    unsigned input_offset = 0, output_offset = round_seq_length / BASE;

    CUDA_CHECK(cudaMemcpy(result_gpu, result, res_length * sizeof(unsigned), cudaMemcpyHostToDevice));
#if DEBUG
    printf("k_cadd loop start\n");
#endif
    for (unsigned kernels = round_seq_length / (BASE * BASE); kernels > 0; kernels /= BASE)
    {
        /**
         * Then, do this addition recursively until there is only one kernel
         * remaining which calculates the total number of non-'-' and non-'\0'
         * characters.
         */
        //TODO: ADD correct kernel parameters
#if DEBUG
    printf("k_cadd loop %d\n", kernels);
#endif
        k_cadd<<<grid1d,block1d>>>(result_gpu, output_offset, input_offset);
        
        input_offset = output_offset;
        output_offset += kernels;
    }
 #if DEBUG
    printf("k_cadd loop end\n");
#endif   
    // Retrieve the result buffer 

 #if DEBUG
    printf("k_cadd loop end\n");
#endif   
    CUDA_CHECK(cudaMemcpy(result, result_gpu, res_length * sizeof(unsigned), cudaMemcpyDeviceToHost));
 #if DEBUG
    printf("k_cadd loop end\n");
#endif

    gettimeofday(&end_time, NULL);
    long gpu_time = (end_time.tv_sec*1000000+end_time.tv_usec) - (start_time.tv_sec*1000000+start_time.tv_usec);

    // Reverse bandwidth intensive stuff goes here

    gettimeofday(&end_time, NULL);
    long bw2_time = (end_time.tv_sec*1000000+end_time.tv_usec) - (start_time.tv_sec*1000000+start_time.tv_usec);


    // CPU intensive stuff goes here
#if DEBUG
    printf("cpu part start\n");
    for (int i=0; i<res_length; ++i)
    {
        printf("%d ", result[i]);
    }
    printf("\n");
#endif
    if (letter_index > result[res_length - 1])
    {
        fprintf(stderr, "Logical index out of bounds (last index: %u).\n", result[res_length - 1]);
	CUDA_CHECK(cudaFree(result_gpu));
    	CUDA_CHECK(cudaFree(seq_gpu));
	exit(-1);
    }

    if (!letter_index)
    {
        fprintf(stderr, "Please used 1-based indexing (for whatever reason).\n");
	CUDA_CHECK(cudaFree(result_gpu));
	CUDA_CHECK(cudaFree(seq_gpu));
	exit(-1);
    }

    /**
     * Okay, now we have a buffer which contains a tree of sums, looking
     * something like this:
     *                  _
     *        4          |
     *      /   \        |
     *    3       1      |- result buffer
     *   / \     / \     |
     *  2   1   1   0   _|
     * / \ / \ / \ / \
     * A G - T C - - -  --- sequence buffer
     *
     * (actually, it looks more like 2 1 1 2 3 3 6)
     *
     * Now, we walk through it from the top. Let's assume we're looking for the
     * logical index 2. We'll compare it to 4: Of course, it's smaller (that was
     * the assertition right before this comment), else, we'd be out of bounds.
     * No we're comparing it with the left 3 in the next level. It's smaller,
     * therefore, this subtree is correct and we move on to the next level.
     * There, we compare it to the left 2. 2 is greater/equal to 2, therefore,
     * this is _not_ the right subtree, we have to go to the other one (the one
     * to the right, below the 1). We subtract the 2 from the left subtree,
     * therefore our new "local" index is 0 (we're looking for the nucleobase at
     * index 0 in the subtree below the 1). Now, at the sequence level, there
     * are always just two possibilities. Either, the local index is 0 or it is
     * 1. If it's 1, this will always mean the right nucleobase, since 1 means
     * to skip one. The only one to skip is the left one, therefore, the right
     * one is the one we're looking for. If the local index is 0, this refers to
     * the first nucleobase, which may be either the left or the right,
     * depending on whether the left one is actually a nucleobase.
     *
     * In this case, the local index is 0. Since the left nucleobase is not
     * really one (it is '-'), the right one is the one we're looking for; its
     * index in the sequence buffer is 3.
     *
     * The reference implementation seems to go total hazels, since it
     * apparently uses 1-based indexing. Logical index 2 would refer to G for
     * it, therefore it returns 2 (which is the 1-based index of G in the
     * sequence buffer). I can't see it from the code, but that is what the
     * result is.
     *
     *
     * For another BASE than 2, it looks like this (BASE 4):
     *
     *                9
     *        //             \\
     *    3       1       3       2
     *  // \\   // \\   // \\   // \\
     * A G - T C - - - C - T T A G - -
     *
     * Let's assume, we're looking for index 5. Compare it to 9, it's smaller,
     * so this is the tree we're looking for. Then compare it to all subtrees:
     * 5 is greater than 3, so go right and subtract 3 from 5. 2 is greater than
     * 1, so go right and subtract 1 from 2. 1 then is smaller than 3, so the
     * third subtree from the left is the one we want to enter now. The index 1
     * here refers to the first T, therefore, it is globally the second T in the
     * sequence.
     */

    // "Current level subtree starting index"; index of the first subtree sum in
    // the current level (we skip level 0, i.e., the complete tree)
    clstsi = res_length - 1 - BASE;
    // "Current level subtree count"; number of subtrees in the current level
    clstc = BASE;
    // "Current level subtree offset"; index difference of the actual set of
    // subtrees we're using from the first one in the current level
    clsto = 0;
    // Turn 1-based index into 0-based
    local_index = letter_index - 1;

    for (;;)
    {
        int subtree;

        // "First subtree index", index of the first subtree we're supposed to
        // examine
        unsigned fsti = clstsi + clsto * BASE;

        // We could add a condition (subtree < BASE) to this loop, but this loop
        // has to be left before this condition is false anyway (otherwise,
        // something is very wrong).
        for (subtree = 0; local_index >= result[fsti + subtree]; subtree++)
            local_index -= result[fsti + subtree];

        // And we'll check it here anyway (#ifdef NDEBUG).
        assert(subtree < BASE);


        clsto = clsto * BASE + subtree;

        // If clstsi is 0, we were at the beginning of the result buffer and are
        // therefore finished
        if (!clstsi)
            break;

        clstc *= BASE;
        clstsi -= clstc;
    }

    // Now we need to go to the sequence level which requires an extra step.
    unsigned index;
    for (index = clsto * BASE; local_index; index++)
        if (sequence[index] != '-')
            local_index--;

    /*** END OF ROCKET SCIENCE LEVEL RUNTIME-TIME INTENSIVE STUFF ***/

    //replace with gettimeofday for CUDA
    gettimeofday(&end_time, NULL);
    delta_time = (end_time.tv_sec*1000000+end_time.tv_usec) - (start_time.tv_sec*1000000+start_time.tv_usec);
    printf("%li us elapsed total\n", delta_time);
    printf(" - %li us on bandwidth forth\n", bw1_time);
    printf(" - %li us on GPU\n", gpu_time - bw1_time);
    printf(" - %li us on bandwidth back\n", bw2_time - gpu_time);
    printf(" - %li us on CPU\n", delta_time - bw2_time);

    printf("Index for %u: %u\n", letter_index, index);
    printf("cnt = %u (index + 1)\n", index + 1);


    //free resource
    CUDA_CHECK(cudaFree(result_gpu));
    CUDA_CHECK(cudaFree(seq_gpu));

    return 0;
}
