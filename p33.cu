#include <string>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include "input_image_cuda.h"
#include "complex_cuda.cc"

__global__ void calculateRow(Complex *copyArr, Complex *holdArr, int *wid, int *numThreads) {
	int index = blockIdx.x * *numThreads + threadIdx.x;
	int rowIndex = index / *wid;
	int n = index % *wid;
	Complex c;
	Complex calcValue, holdValue;

	__syncthreads();
	for (int k = 0; k < *wid; k++) {
		c = Complex(cos(2 * PI * k * n / *wid), -sin(2 * PI * k * n / *wid));
		calcValue =  c * holdArr[k + (rowIndex * *wid)];
		holdValue = holdValue + calcValue;
	}
	copyArr[index] = holdValue;
	__syncthreads();
}

__global__ void calculateColumn(Complex *copyArr, Complex *holdArr, int *hei, int *wid, int *numThreads) {
	int index = blockIdx.x * *numThreads + threadIdx.x;
	int colIndex = index % *hei;
	int n = index / *hei;
	Complex c;
	Complex calcValue, holdValue;

	__syncthreads();
	for (int k = 0; k < *hei; k++) {
		c = Complex(cos(2 * PI * k * n / *hei), -sin(2 * PI * k * n / *hei));
		calcValue =  c * holdArr[(k * *wid) + colIndex];
		holdValue = holdValue + calcValue;
	}
	copyArr[index] = holdValue;
	__syncthreads();
}

int main(int argc, char *argv[]) {

	std::string inFile;
	inFile = std::string(argv[2]);
	
	InputImage imgFile(inFile.c_str());
	Complex *arr = imgFile.get_image_data();
	int width = imgFile.get_width();
	int height = imgFile.get_height();
	int sizeArr = width * height;

	int threads, blocks;

	if (sizeArr % 32 == 0) {
		threads = 32;
		blocks = sizeArr / threads;
	}
	else {
		for (int a = 31; a > 1; a--) {
			if (sizeArr % a == 0) {
				threads = a;
				blocks = sizeArr / threads;
				a = 1;
			}
			else if (a == 2 && sizeArr % a != 0) {
				threads = 1;
				blocks = sizeArr;
			}
		}
	}

	Complex *copyArr, *holdArr;
	int *wid, *hei, *numThreads;

	int memSize = sizeArr * sizeof(Complex);
	int intSize = sizeof(int);

	copyArr = (Complex *) malloc(memSize);
	holdArr = (Complex *) malloc(memSize);
	wid = (int *) malloc(intSize);
	hei = (int *) malloc(intSize);
	numThreads = (int *) malloc(intSize);

	cudaMalloc((void **)&copyArr, memSize);
	cudaMalloc((void **)&holdArr, memSize);
	cudaMalloc((void **)&wid, intSize);
	cudaMalloc((void **)&hei, intSize);
	cudaMalloc((void **)&numThreads, intSize);

	cudaMemcpy(copyArr, arr, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(holdArr, arr, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(wid, &width, intSize, cudaMemcpyHostToDevice);
	cudaMemcpy(hei, &height, intSize, cudaMemcpyHostToDevice);
	cudaMemcpy(numThreads, &threads, intSize, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	calculateRow<<<blocks, threads>>> (copyArr, holdArr, wid, numThreads);

	cudaDeviceSynchronize();

	cudaMemcpy(arr, copyArr, memSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(holdArr, arr, memSize, cudaMemcpyHostToDevice);

	calculateColumn<<<blocks, threads>>> (copyArr, holdArr, hei, wid, numThreads);

	cudaDeviceSynchronize();

	cudaMemcpy(arr, copyArr, memSize, cudaMemcpyDeviceToHost);

	std::string outFile;
	outFile = std::string(argv[3]);
	imgFile.save_image_data(outFile.c_str(), arr, width, height);

	cudaFree(copyArr);
	cudaFree(holdArr);
	cudaFree(wid);
	cudaFree(hei);
	cudaFree(numThreads);

	return 0;
}