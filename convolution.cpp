

void host_convolution(const uchar *M, const float *K, uchar *out, const int H, const int W){   

    int ker_r = KER/2;
    for(int i = 0; i < W; i++){
        for(int j = 0; j < H; j++){
            for(int k = 0; k < KER; k++){
                for(int l = 0; l < KER; l++){
                    out[i*W+j] += (uchar)M[(i-ker_r+k)*W+(j-ker_r+l)]*K[k*KER+l];
                }
            }
            }
        }
}
