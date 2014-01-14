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
 * Returns the integer logarithm to 2^exp (round down).
 */
static int log_pow2(long x, int exp)
{
    assert(x > 0);

    int log;
    for (log = -1; x; log++)
        x >>= exp;

    return log;
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
    char *sequence = load_text(argv[argc - 1], &seq_length);
    if (!sequence)
        return 1;

    seq_length--; // Cut final 0 byte

    // FIXME: All the following code relies on seq_length being a multiple of BASE.

    long round_seq_length = round_up_to_power_of_two(seq_length, BASE_EXP);
    int levels = log_pow2(round_seq_length, BASE_EXP);


    long subtree_offsets[levels + 1];
    long subtrees[levels];
    long current_offset = 0, current_subtrees = (seq_length + BASE - 1) / BASE;
    for (int level = 0; level <= levels; level++)
    {
        subtree_offsets[level] = current_offset;

        if (level < levels)
            subtrees[level] = current_subtrees;

        current_offset += current_subtrees;
        current_subtrees = (current_subtrees + BASE - 1) / BASE;
    }

    assert(current_subtrees == 1);

    long res_length = subtree_offsets[levels];


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
    char *prog_src = load_text("trans.cl", NULL);
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
    unsigned *result = malloc(res_length * sizeof(unsigned));
    cl_mem result_gpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, res_length * sizeof(unsigned), result, NULL);


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
    clSetKernelArg(k_iadd, 2, sizeof(unsigned), &(unsigned){seq_length});
    clEnqueueNDRangeKernel(queue, k_iadd, 1, NULL, &(size_t){subtrees[0]}, NULL, 0, NULL, NULL);

    clSetKernelArg(k_cadd, 0, sizeof(result_gpu), &result_gpu);

    for (int level = 1; level < levels; level++)
    {
        /**
         * Then, do this addition recursively until there is only one kernel
         * remaining which calculates the total number of non-'-' and non-'\0'
         * characters.
         */
        if (subtrees[level] > 0)
        {
            clSetKernelArg(k_cadd, 1, sizeof(unsigned), &(unsigned){subtree_offsets[level]});
            clSetKernelArg(k_cadd, 2, sizeof(unsigned), &(unsigned){subtree_offsets[level - 1]});
            clFinish(queue);
            clEnqueueNDRangeKernel(queue, k_cadd, 1, NULL, &(size_t){subtrees[level] - 1}, NULL, 0, NULL, NULL);
        }

        // Execute the last kernel on the CPU
        {
            unsigned in_start = subtree_offsets[level - 1] + ((subtrees[level] - 1) << BASE_EXP);
            unsigned out_pos = subtree_offsets[level] + subtrees[level] - 1;
            unsigned result_val = 0;

            for (unsigned i = in_start; i < subtree_offsets[level]; i++)
                result_val += result[i];

            result[out_pos] = result_val;
        }
    }

    clFinish(queue);

    long gpu_time = clock_delta();


    // Reverse bandwidth intensive stuff goes here

    // Retrieve the result buffer
    // clEnqueueReadBuffer(queue, result_gpu, true, 0, res_length * sizeof(unsigned), result, 0, NULL, NULL);
    // (No need to do that, since we're fixed on CL_MEM_USE_HOST_PTR here)

    long bw2_time = clock_delta();


    // CPU intensive stuff goes here

    if (letter_index > result[res_length - 1])
    {
        fprintf(stderr, "Logical index out of bounds (last index: %u).\n", result[res_length - 1]);
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

    int level = levels - 1;
    // "Current level subtree offset"; index difference of the actual set of
    // subtrees we're using from the first one in the current level
    unsigned clsto = 0;
    // Turn 1-based index into 0-based
    unsigned local_index = letter_index - 1;

    for (;;)
    {
        int subtree;

        // "First subtree index", index of the first subtree we're supposed to
        // examine
        unsigned fsti = subtree_offsets[level] + clsto * BASE;

        for (subtree = 0; local_index >= result[fsti + subtree]; subtree++)
            local_index -= result[fsti + subtree];

        assert(subtree < subtrees[level]);


        clsto = clsto * BASE + subtree;

        if (!level--)
            break;
    }

    // Now we need to go to the sequence level which requires an extra step.
    unsigned index;
    for (index = clsto * BASE; local_index; index++)
        if (sequence[index] != '-')
            local_index--;

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
