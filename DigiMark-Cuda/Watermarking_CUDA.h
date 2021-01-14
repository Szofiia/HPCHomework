// Helper Header File
#pragma once
#ifndef _Watermarking_CUDA_
#define _Watermarking_CUDA_
void CalcRandWithHostAPI(float* host_data, int N);
void CalcRandWithDevAPI(float* host_data, int N);
void CalcWatermark(float* host_data, int N, float w, float alpha);
#endif