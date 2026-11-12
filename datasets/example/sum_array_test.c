/* Test driver: includes the sample, calls the function as its callers do, and
 * exits 0 only if every check holds. */
#include "sum_array.c"

int main(void)
{
    int a[] = {1, 2, 3, 4};
    int b[] = {-5, 5, 7};
    if (sum_array(a, 4) != 10) return 1;
    if (sum_array(a, 0) != 0) return 2;
    if (sum_array(b, 3) != 7) return 3;
    return 0;
}
