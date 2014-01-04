#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#include "common.h"

extern void clock_start(void);
extern long clock_delta(void);


long round_up_to_power_of_two(long x, int exp)
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


char *load_text(const char *filename, long *length_ptr, int round_exp)
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
    if (argc < 2)
    {
        fprintf(stderr, "Usage: transalign_killer [options] <input file>\n");
        return 1;
    }


    long seq_length;
    /* The exponent given here determines the steps taken in the adding kernel. An exponent of 1 results in rounding the
     * size to 2^1 = 2, therefore, in every step, two input fields are added and the size shrinks to half of what it was
     * before. */
    char *sequence = load_text(argv[argc - 1], &seq_length, 1);
    if (!sequence)
        return 1;


    unsigned letter_index = seq_length / 2;


    cl_device_id dev = select_device(argc - 1, argv);
    if (!dev)
        return 1;

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev, 0, NULL);

    char *prog_src = load_text("trans.cl", NULL, 0);
    if (!prog_src)
        return 1;
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char **)&prog_src, NULL, NULL);
    free(prog_src);

    clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
    cl_kernel k_iadd = clCreateKernel(prog, "k_iadd", NULL);
    cl_kernel k_cadd = clCreateKernel(prog, "k_cadd", NULL);
    assert(k_iadd);
    assert(k_cadd);


    cl_mem result_gpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE, seq_length * sizeof(unsigned), NULL, NULL);
    unsigned *result = malloc(seq_length * sizeof(unsigned));


    clock_start();

    /*** START OF ROCKET SCIENCE LEVEL RUNTIME-TIME INTENSIVE STUFF ***/

    // Bandwidth intensive stuff goes here

    cl_mem seq_gpu = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, seq_length * sizeof(char), sequence, NULL);

    long bw1_time = clock_delta();


    // GPU intensive stuff goes here

    clSetKernelArg(k_iadd, 0, sizeof(result_gpu), &result_gpu);
    clSetKernelArg(k_iadd, 1, sizeof(seq_gpu), &seq_gpu);
    clEnqueueNDRangeKernel(queue, k_iadd, 1, NULL, &(size_t){seq_length / 2}, NULL, 0, NULL, NULL);

    for (unsigned kernels = seq_length / 4; kernels > 0; kernels /= 2)
    {
        clSetKernelArg(k_cadd, 0, sizeof(result_gpu), &result_gpu);
        clSetKernelArg(k_cadd, 1, sizeof(unsigned), &(unsigned){seq_length - kernels * 2});
        clSetKernelArg(k_cadd, 2, sizeof(unsigned), &(unsigned){seq_length - kernels * 4});
        clFinish(queue);
        clEnqueueNDRangeKernel(queue, k_cadd, 1, NULL, &(size_t){kernels}, NULL, 0, NULL, NULL);
    }

    clFinish(queue);

    long gpu_time = clock_delta();


    // Reverse bandwidth intensive stuff goes here

    clEnqueueReadBuffer(queue, result_gpu, true, 0, seq_length * sizeof(unsigned), result, 0, NULL, NULL);

    long bw2_time = clock_delta();


    // CPU intensive stuff goes here

    assert(letter_index < result[seq_length - 2]);

    unsigned index = seq_length - 2;
    unsigned compare = letter_index;
    while (index >= seq_length / 2)
    {
        bool go_right = compare >= result[index];
        if (go_right)
            compare -= result[index];

        index = ((index | go_right) << 1) & (seq_length - 1);
    }

    bool go_right = compare >= result[index];
    if (go_right)
        compare -= result[index];
    assert(compare <= 1);

    index = ((index | go_right) << 1) + compare;

    /*** END OF ROCKET SCIENCE LEVEL RUNTIME-TIME INTENSIVE STUFF ***/

    long delta_time = clock_delta();
    printf("%li us elapsed total\n", delta_time);
    printf(" - %li us on bandwidth forth\n", bw1_time);
    printf(" - %li us on GPU\n", gpu_time - bw1_time);
    printf(" - %li us on bandwidth back\n", bw2_time - gpu_time);
    printf(" - %li us on CPU\n", delta_time - bw2_time);

    printf("Index for %u: %u\n", letter_index, index);
    printf("cnt = %u (index - 1)\n", index - 1);


    free(sequence);
    free(result);

    clReleaseMemObject(result_gpu);
    clReleaseMemObject(seq_gpu);
    clReleaseKernel(k_iadd);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
}
