#include <string.h>

void reverse_string(char *text)
{
    size_t len = strlen(text);
    for (size_t i = 0; i < len / 2; i++) {
        char tmp = text[i];
        text[i] = text[len - 1 - i];
        text[len - 1 - i] = tmp;
    }
}
