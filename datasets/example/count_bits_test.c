#include "count_bits.c"

int main(void)
{
    if (count_bits(0) != 0) return 1;
    if (count_bits(1) != 1) return 2;
    if (count_bits(0xF0F0) != 8) return 3;
    if (count_bits(0xFFFFFFFFFFFFFFFFull) != 64) return 4;
    if (count_bits(0x8000000000000000ull) != 1) return 5;
    return 0;
}
