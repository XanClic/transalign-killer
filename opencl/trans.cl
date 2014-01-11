#define BASE_EXP 4
#define BASE (1 << BASE_EXP)

// initial addition
__kernel void k_iadd(__global unsigned *dest, __global char *sequence, unsigned seq_length)
{
    unsigned id = get_global_id(0);
    unsigned in_start = id << BASE_EXP;
    unsigned result = 0;

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

// consecutive addition
__kernel void k_cadd(__global unsigned *buffer, unsigned doff, unsigned soff)
{
    unsigned id = get_global_id(0);
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
