#include <stdint.h>

unsigned count_bits(uint64_t word)
{
    unsigned bits = 0;
    while (word) {
        word &= word - 1;
        bits++;
    }
    return bits;
}
