#include "SIMD.h"

#include <memory.h>
#include <cstdio>
#include <fstream>


u32 IF(u32 A, u32 B, u32 C) {
    return ((A & B) | (~A & C));
}

u32 MAJ(u32 A, u32 B, u32 C) {
    return ((A & B) | (A & C) | (B & C));
}


SIMD::SIMD(unsigned char* M, unsigned int size) {
    data = M;
    databitlen = size;
    run();
}

SIMD::SIMD(const std::string &filePath, unsigned int size) {
    readFile(filePath);
    databitlen = size;
    run();
}

SIMD::SIMD(const std::string &filePath) {
    readFile(filePath);
    run();
}

void SIMD::readFile(const std::string &filePath) {
    std::ifstream fs(filePath, std::ios_base::out | std::ios_base::binary);
    if (fs.fail())
        exit(1);

    fs.seekg(0, std::ios::end);
    auto fileSize = static_cast<unsigned long>(fs.tellg());
    fs.seekg(0, std::ios::beg);

    data = new unsigned char[fileSize];
    databitlen = fileSize * 8;

    fs.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(fileSize));
}

void SIMD::run() {
    Init();
    Update();
    Final();

    printResult();
    delete[] A;
    delete[] buffer;
}

void SIMD::Init() {
    blocksize = 512;
    count = 0;

    buffer = new unsigned char[64];

    A = new u32[16];
    B = A + 4;
    C = B + 4;
    D = C + 4;
    memset(A, 0, 64);

    auto *IV = new unsigned char[64];
    memset(IV, 0, 64);
    snprintf((char*)IV, 64, "SIMD-256 v1.1");

    SIMD_Compress(IV, 0);
    delete[] IV;
}

void SIMD::Update() {

    while (databitlen > 0) {
        if (databitlen >= blocksize) {
            SIMD_Compress(data, 0);
            databitlen -= blocksize;
            data += blocksize/8;
            count += blocksize;
        }
        else {
            unsigned int len = blocksize;
            if (databitlen < len) {
                memcpy(buffer, data, (databitlen+7)/8);
                count += databitlen;
                return;
            }
            else {
                memcpy(buffer, data, len/8);
                count += len;
                databitlen -= len;
                data += len/8;
                SIMD_Compress(buffer, 0);
            }
        }
    }
}

void SIMD::Final() {
    int current = count & (blocksize - 1);

    if (current) {
        if (current & 7) {
            auto mask = static_cast<unsigned char>(0xff >> (current & 7));
            buffer[current/8] &= ~mask;
        }
        current = (current+7)/8;
        memset(buffer+current, 0, blocksize / 8 - current);
        SIMD_Compress(buffer, 0);
    }

    memset(buffer, 0, blocksize / 8);

    int l = count;
    for (int i = 0; i < 8; i++) {
        buffer[i] = static_cast<unsigned char>(l & 0xff);
        l >>= 8;
    }

    SIMD_Compress(buffer, 1);

    unsigned char bs[64];
    for (int i = 0; i < 8; i++) {
        unsigned int x = A[i];
        bs[4*i  ] = static_cast<unsigned char>(x & 0xff);
        x >>= 8;
        bs[4*i+1] = static_cast<unsigned char>(x & 0xff);
        x >>= 8;
        bs[4*i+2] = static_cast<unsigned char>(x & 0xff);
        x >>= 8;
        bs[4*i+3] = static_cast<unsigned char>(x & 0xff);
    }

    result = new unsigned char[32];
    memcpy(result, bs, 32);
}

void SIMD::SIMD_Compress(const unsigned char *M, int final) {
    u32 W[32][4];
    u32 IV[4][4];
    const int n = 4;

    for(int i = 0; i < n; i++) {
        IV[0][i] = A[i];
        IV[1][i] = B[i];
        IV[2][i] = C[i];
        IV[3][i] = D[i];
    }

    message_expansion(W, M, final);

    for(int j = 0; j < n; j++) {
        A[j] ^= (M[4*j] ^ ((M[4*j+1]) << 8) ^ ((M[4*j+2]) << 16) ^ ((M[4*j+3]) << 24));
        B[j] ^= (M[4*j+4*n] ^ ((M[4*j+4*n+1]) << 8) ^ ((M[4*j+4*n+2]) << 16) ^ ((M[4*j+4*n+3]) << 24));
        C[j] ^= (M[4*j+8*n] ^ ((M[4*j+8*n+1]) << 8) ^ ((M[4*j+8*n+2]) << 16) ^ ((M[4*j+8*n+3]) << 24));
        D[j] ^= (M[4*j+12*n] ^ ((M[4*j+12*n+1]) << 8) ^ ((M[4*j+12*n+2]) << 16) ^ ((M[4*j+12*n+3]) << 24));
    }

    Round(W, 0, 3,  23, 17, 27);
    Round(W, 1, 28, 19, 22,  7);
    Round(W, 2, 29,  9, 15,  5);
    Round(W, 3,  4, 13, 10, 25);

    Step(IV[0], 32, 4,  13, IF);
    Step(IV[1], 33, 13, 10, IF);
    Step(IV[2], 34, 10, 25, IF);
    Step(IV[3], 35, 25,  4, IF);
}

void SIMD::message_expansion(u32 W[32][4], const unsigned char *const M, int final) {
    const int P[32] = {4, 6, 0, 2,  7, 5, 3, 1,
                       15,11,12,8,  9, 13,10,14,
                       17,18,23,20, 22,21,16,19,
                       30,24,25,31, 27,29,28,26};

    short y[256];

    // FFT of y_i
    const int alpha = 139;
    int beta = 98;  // alpha^127
    int beta_i = 1;
    int alpha_i = 1;
    u32 Z[32][4];

    const int fft_size = 128;
    const int M_size = 64;

    for(int i=0; i<fft_size; i++) {
        y[i] = static_cast<short>(beta_i);
        beta_i = (beta_i * beta) % 257;
    }


    if (final) {
        beta = 58;
        beta_i = 1;
        for(int i=0; i<fft_size; i++) {
            y[i] += beta_i;
            beta_i = (beta_i * beta) % 257;
        }
    }


    for(int i=0; i<fft_size; i++) {
        int alpha_ij = 1; // alpha^(i*j)

        for(int j=0; j<M_size; j++) {

            y[i] = static_cast<short>((y[i] + alpha_ij * M[j]) % 257);
            alpha_ij = (alpha_ij * alpha_i) % 257;
        }

        alpha_i = (alpha_i * alpha) % 257;
    }


    for(int i=0; i<fft_size; i++)
        if (y[i] > 128)
            y[i] -= 257;

    for(int i=0; i<16; i++)
        for(int j=0; j<4; j++)
            Z[i][j] = (((u32) (y[2*i*4+2*j] * 185)) & 0xffff) | ((u32) (y[2*i*4+2*j+1] * 185) << 16);

    for(int i=0; i<8; i++)
        for(int j=0; j<4; j++)
            Z[i+16][j] = (((u32) (y[2*i*4+2*j] * 233)) & 0xffff) | ((u32) (y[2*i*4+2*j+fft_size/2] * 233) << 16);

    for(int i=0; i<8; i++)
        for(int j=0; j<4; j++)
            Z[i+24][j] = (((u32) (y[2*i*4+2*j+1] * 233)) & 0xffff) | ((u32) (y[2*i*4+2*j+fft_size/2+1] * 233) << 16);


    for(int i=0; i<32; i++)
        for(int j=0; j<4; j++)
            W[i][j] = Z[P[i]][j];
}

void SIMD::Round(u32 w[32][4], const int i, const int p1, const int p2, const int p3, const int p4) {
    Step(w[8*i],   8*i,   p1, p2, IF);
    Step(w[8*i+1], 8*i+1, p2, p3, IF);
    Step(w[8*i+2], 8*i+2, p3, p4, IF);
    Step(w[8*i+3], 8*i+3, p4, p1, IF);

    Step(w[8*i+4], 8*i+4, p1, p2, MAJ);
    Step(w[8*i+5], 8*i+5, p2, p3, MAJ);
    Step(w[8*i+6], 8*i+6, p3, p4, MAJ);
    Step(w[8*i+7], 8*i+7, p4, p1, MAJ);
}

void SIMD::Step(const u32 *w, const int i, const int r, const int s, const bool_func F) {
    int p[][8] = {{1,0,3,2},
                  {2,3,0,1},
                  {3,2,1,0}};

    u32 tmp[8];
    for(int j = 0; j < 4; j++)
        tmp[j] = ROTL32(A[j], r);

    for(int j = 0; j < 4; j++) {
        A[j] = D[j] + w[j] + F(A[j], B[j], C[j]);
        A[j] = ROTL32(A[j], s) + tmp[p[i % 3][j]];
        D[j] = C[j];
        C[j] = B[j];
        B[j] = tmp[j];
    }
}

void SIMD::printResult() {
    for (int i = 0; i < 32; i++)
        printf ("%02x", result[i]);
    printf("\n");
}
