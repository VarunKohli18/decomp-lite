#include "clamp_scale.c"

int main(void)
{
    if (clamp_scale(0.25, 2.0) != 0.5) return 1;
    if (clamp_scale(3.0, 2.0) != 1.0) return 2;
    if (clamp_scale(-3.0, 2.0) != -1.0) return 3;
    if (clamp_scale(0.0, 100.0) != 0.0) return 4;
    return 0;
}
