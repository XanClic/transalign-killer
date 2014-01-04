// initial addition
__kernel void k_iadd(__global unsigned *dest, __global char *sequence)
{
    unsigned i = get_global_id(0);
    char a = sequence[i * 2 + 0];
    char b = sequence[i * 2 + 1];

    dest[i] =
        ((a != '-') && a) +
        ((b != '-') && b);
}

// consecutive addition
__kernel void k_cadd(__global unsigned *buffer, unsigned doff, unsigned soff)
{
    unsigned i = get_global_id(0);

    buffer[doff + i] =
        buffer[soff + i * 2 + 0] +
        buffer[soff + i * 2 + 1];
}
