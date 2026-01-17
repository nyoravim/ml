#include <nyoravim/base64.h>
#include <nyoravim/mem.h>

#include "data/mnist.h"

int main(int argc, const char** argv) {
    struct mnist* data = mnist_load("data/train-images-idx3-ubyte.gz");
    if (!data) {
        return 1;
    }

    mnist_free(data);
    return 0;
}
