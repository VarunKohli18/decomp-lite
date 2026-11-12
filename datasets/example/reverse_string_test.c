#include "reverse_string.c"

int main(void)
{
    char a[] = "hello", b[] = "", c[] = "ab", d[] = "x";
    reverse_string(a);
    reverse_string(b);
    reverse_string(c);
    reverse_string(d);
    if (strcmp(a, "olleh")) return 1;
    if (strcmp(b, "")) return 2;
    if (strcmp(c, "ba")) return 3;
    if (strcmp(d, "x")) return 4;
    return 0;
}
