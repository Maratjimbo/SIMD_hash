#ifndef SIMD_H
#define SIMD_H

#include <string>

#define ROTL32(x, n) (0xFFFFFFFF & ((x) << (n)) | ((x) >> (32 - (n))))

typedef unsigned int u32;
typedef u32 (*bool_func) (const u32, const u32, const u32);


class SIMD {
private:
    unsigned char *data;
    unsigned int databitlen;

    unsigned int blocksize;
    int count;

    u32 *A, *B, *C, *D;
    unsigned char* buffer;

    unsigned char* result;

private:
    void readFile(const std::string &filePath);
    void run();
    void printResult();

    void Init();
    void Update();
    void Final();

    void SIMD_Compress(const unsigned char *M, int final);

    void message_expansion(u32 W[32][4], const unsigned char *M, int final);
    void Round(u32 w[32][4], int i, int p1, int p2, int p3, int p4);
    void Step(const u32 w[4], int i, int r, int s, bool_func F);

public:
    SIMD(unsigned char* M, unsigned int size);
    SIMD(const std::string &filePath, unsigned int size);
    explicit SIMD(const std::string &filePath);

};


#endif //SIMD_H

