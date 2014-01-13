#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


extern void clock_start(void);
extern long clock_delta(void);

extern unsigned long find_index(char *sequence, unsigned long logical_index);


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
        fprintf(stderr, "Usage: transalign_killer <input file>\n");
        return 1;
    }


    long seq_length;
    char *sequence = load_text(argv[argc - 1], &seq_length);
    if (!sequence)
        return 1;

    seq_length--; // Cut final 0 byte

    // Use some random index to be searched for here
    unsigned long letter_index = seq_length / 2;

    clock_start();

    unsigned long index = find_index(sequence, letter_index - 1);

    long delta_time = clock_delta();

    printf("%li us elapsed total\n\n", delta_time);

    printf("Index for %lu: %lu\n", letter_index, index);
    printf("cnt = %lu (index + 1)\n", index + 1);


    free(sequence);

    return ret;
}
