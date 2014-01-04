#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#include "common.h"


/* The exponent given here determines the steps taken in the adding kernel. An
 * exponent of 1 results in rounding the size to 2^1 = 2, therefore, in every
 * step, two input fields are added and the size shrinks to half of what it was
 * before. This influences the size of the result buffer as well (the greater
 * this exponent is, the smaller the result will be). */
#define BASE_EXP 1
#define BASE (1 << BASE_EXP)

/* Define this to actually use host memory instead of copying the buffer to the
 * GPU (as it turns out, this may actually be worth it) */
// #define USE_HOST_PTR


#ifdef USE_HOST_PTR
#define HOST_PTR_POLICY CL_MEM_USE_HOST_PTR
#else
#define HOST_PTR_POLICY CL_MEM_COPY_HOST_PTR
#endif


/**
 * These two functions provide std::chrono functionality (see cpp-stuff.cpp for
 * an explanation why they're extern).
 */
extern void clock_start(void);
extern long clock_delta(void);


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
 * Loads a text file and returns a buffer with the contents. If length_ptr is
 * not NULL, the buffer length (in memory!) is stored there. If round_exp is 0,
 * the buffer length will be the file length + 1; if it is greater than 0,
 * round_up_to_power_of_two(file length, round_exp) will be used as the buffer
 * length. The part of the buffer behind the end of file will be set to 0.
 *
 * Therefore, setting round_exp to 0 results in a simple C string.
 */
static char *load_text(const char *filename, long *length_ptr, int round_exp)
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

    long mem_len = round_exp ? round_up_to_power_of_two(length, round_exp) : length + 1;

    if (length_ptr)
        *length_ptr = mem_len;

    char *content = calloc(mem_len, 1);
    fread(content, 1, length, fp);
    fclose(fp);

    return content;
}


int main(int argc, char *argv[])
{
    int ret = 0;


    if (argc < 2)
    {
        fprintf(stderr, "Usage: transalign_killer [--cldev=x.y] <input file>\n");
        fprintf(stderr, "  --cldev=x.y: x specifies the platform index, y the device index.\n");
        return 1;
    }


    long seq_length;
    char *sequence = load_text(argv[argc - 1], &seq_length, BASE_EXP);
    if (!sequence)
        return 1;


    // Use some random index to be searched for here
    unsigned letter_index = seq_length / 2;


    // Select an OpenCL device
    cl_device_id dev = select_device(argc - 1, argv);
    if (!dev)
        return 1;

    // Initialize the OpenCL st...ack
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev, 0, NULL);

    // Load the OpenCL kernesl
    char *prog_src = load_text("trans.cl", NULL, 0);
    if (!prog_src)
        return 1;
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&prog_src, NULL, NULL);
    free(prog_src);

    // Build them
    clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
    cl_kernel k_iadd = clCreateKernel(prog, "k_iadd", NULL); // initial addition
    cl_kernel k_cadd = clCreateKernel(prog, "k_cadd", NULL); // consecutive addition
    assert(k_iadd);
    assert(k_cadd);


    // Create the result buffer
    unsigned *result = malloc(seq_length * sizeof(unsigned));
    cl_mem result_gpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE | HOST_PTR_POLICY, seq_length * sizeof(unsigned), result, NULL);


    clock_start();

    /*** START OF ROCKET SCIENCE LEVEL RUNTIME-TIME INTENSIVE STUFF ***/

    // Bandwidth intensive stuff goes here

    // Copy the sequence to the video memory (or, generally speaking, the OpenCL device)
    cl_mem seq_gpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE | HOST_PTR_POLICY, seq_length * sizeof(char), sequence, NULL);

    long bw1_time = clock_delta();


    // GPU intensive stuff goes here

    /**
     * First, transform every - and \0 into a 0 and every other character into a
     * 1. Then, add consecutive fields (BASE fields) together and store them at
     * the beginning of the result buffer.
     */
    clSetKernelArg(k_iadd, 0, sizeof(result_gpu), &result_gpu);
    clSetKernelArg(k_iadd, 1, sizeof(seq_gpu), &seq_gpu);
    clEnqueueNDRangeKernel(queue, k_iadd, 1, NULL, &(size_t){seq_length / BASE}, NULL, 0, NULL, NULL);

    for (unsigned kernels = seq_length / (BASE * BASE); kernels > 0; kernels /= BASE)
    {
        /**
         * Then, do this addition recursively until there is only one kernel
         * remaining which calculates the total number of non-'-' and non-'\0'
         * characters.
         */
        clSetKernelArg(k_cadd, 0, sizeof(result_gpu), &result_gpu);
        clSetKernelArg(k_cadd, 1, sizeof(unsigned), &(unsigned){seq_length - kernels * BASE});
        clSetKernelArg(k_cadd, 2, sizeof(unsigned), &(unsigned){seq_length - kernels * BASE * BASE});
        clFinish(queue);
        clEnqueueNDRangeKernel(queue, k_cadd, 1, NULL, &(size_t){kernels}, NULL, 0, NULL, NULL);
    }

    clFinish(queue);

    long gpu_time = clock_delta();


    // Reverse bandwidth intensive stuff goes here

    // Retrieve the result buffer
    clEnqueueReadBuffer(queue, result_gpu, true, 0, seq_length * sizeof(unsigned), result, 0, NULL, NULL);

    long bw2_time = clock_delta();


    // CPU intensive stuff goes here

    if (letter_index > result[seq_length - 2])
    {
        fprintf(stderr, "Logical index out of bounds (last index: %u).\n", result[seq_length - 2]);
        ret = 1;
        goto fail;
    }

    if (!letter_index)
    {
        fprintf(stderr, "Please used 1-based indexing (for whatever reason).\n");
        ret = 1;
        goto fail;
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
     */

    unsigned index = seq_length - 4;
    unsigned local_index = letter_index - 1; // Turn 1-based index into 0-based.
    while (index >= seq_length / BASE)
    {
        /* Is this (the left subtree) actually the correct subtree? If
         * go_right == true, it isn't; subtract the root of the left subtree in
         * this case and go the right one */
        bool go_right = local_index >= result[index];
        if (go_right)
            local_index -= result[index];

        /* Great formula for determining the next index (& (seq_length - 1) is
         * equal to % seq_length, since seq_length needs to be a power of two);
         * don't try to understand it, it just seems to work */
        index = ((index | go_right) << BASE_EXP) & (seq_length - 1);
    }

    // Now we need to go to the sequence level which requires an extra step.
    bool go_right = local_index >= result[index];
    if (go_right)
        local_index -= result[index];
    assert(local_index <= 1);

    index = ((index | go_right) << BASE_EXP) + local_index;

    // See the comment on the local index being 0 at the sequence level above
    if (!local_index && (!sequence[index] || (sequence[index] == '-')))
        index++;

    /*** END OF ROCKET SCIENCE LEVEL RUNTIME-TIME INTENSIVE STUFF ***/

    long delta_time = clock_delta();
    printf("%li us elapsed total\n", delta_time);
    printf(" - %li us on bandwidth forth\n", bw1_time);
    printf(" - %li us on GPU\n", gpu_time - bw1_time);
    printf(" - %li us on bandwidth back\n", bw2_time - gpu_time);
    printf(" - %li us on CPU\n", delta_time - bw2_time);

    printf("Index for %u: %u\n", letter_index, index);
    printf("cnt = %u (index + 1)\n", index + 1);


fail:
    free(sequence);
    free(result);

    clReleaseMemObject(result_gpu);
    clReleaseMemObject(seq_gpu);
    clReleaseKernel(k_iadd);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return ret;
}
