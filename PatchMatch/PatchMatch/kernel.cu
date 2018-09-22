
#include "cuda.h"
#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"


#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <fstream>

#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define PI 3.1415926f


texture<uchar, cudaTextureType2D, cudaReadModeElementType> leftTex;
texture<uchar, cudaTextureType2D, cudaReadModeElementType> rightTex;


__global__ void rand_init(curandState *d_states, int height, int width)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = x + y*width;
	curand_init(1234ULL, offset, 0, &d_states[offset]);
}


__global__ void wrap_phase_shift(uchar* src, float* dst, int height, int width, float diffT)
{
	int imgSize = height*width;
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = x + y*width;

	float sqrt3 = sqrtf(3.0f);
	float I1 = static_cast<float>(src[offset]);
	float I2 = static_cast<float>(src[imgSize + offset]);
	float I3 = static_cast<float>(src[2 * imgSize + offset]);

	float maxI = fmaxf(fmaxf(I1, I2), I3);
	float I1_I2 = fabs(I1 - I2);
	float I2_I3 = fabs(I2 - I3);
	float I1_I3 = fabs(I1 - I3);

	if ((I1_I2 < diffT) && (I2_I3 < diffT) && (I1_I3 < diffT))
	{
		dst[offset] = -4.0f;
		return;
	}
	float phiVal = atan2f(sqrt3*(I1 - I3), (2 * I2 - I1 - I3));
	if (phiVal < 0) phiVal += 2 * PI;
	dst[offset] = phiVal;
}



__global__ void mean_filter(uchar* d_dst1, uchar* d_dst2, uchar* src1, uchar *src2, int height, int width, int win1, int win2)
{
	int winSize = (2 * win1 + 1)*(2 * win2 + 1);
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = x + y*width;
	float sum1 = 0;
	float sum2 = 0;
	for (int h = -win1; h <= win1; h++)
	{
		int y0 = y + h;
		if (y0 < 0) y0 = 0;
		if (y0 >= height) y0 = height - 1;
		for (int w = -win2; w <= win2; w++)
		{
			//sum1 += tex2D(leftTex, x + w, y + h);
			//sum2 += tex2D(rightTex, x + w, y + h);
			int x0 = x + w;
			if (x0 < 0) x0 = 0;
			if (x0 >= width - 1) x0 = width - 1;
			sum1 += src1[y0*width + x0];
			sum2 += src2[y0*width + x0];
		}
	}
	d_dst1[offset] = static_cast<uchar>(sum1 / winSize + 0.5f);
	d_dst2[offset] = static_cast<uchar>(sum2 / winSize + 0.5f);
}


//census transform
__global__ void census_transform64(uchar* d_leftMean, uchar* d_rightMean,
	uint64_t* d_leftCen, uint64_t* d_rightCen, int height, int width, int win1, int win2)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = x + y*width;
	uint64_t leftRes = 0;
	uint64_t rightRes = 0;
	int count = 0;
	for (int h = -win1; h <= win1; h++)
	{
		for (int w = -win2; w <= win2; w++)
		{
			if (h == 0 && w == 0) continue;
			uchar leftTemp = tex2D(leftTex, x + w, y + h);
			uchar rightTemp = tex2D(rightTex, x + w, y + h);
			if (d_leftMean[offset] > leftTemp)
			{
				leftRes = leftRes | (1 << count);
			}
			if (d_rightMean[offset] > rightTemp)
			{
				rightRes = rightRes | (1 << count);
			}
			count++;
		}
	}
	d_leftCen[offset] = leftRes;
	d_rightCen[offset] = rightRes;
}

__global__ void census_transform32(uint32_t *d_leftCen, uint32_t *d_rightCen, uchar *d_leftMean, uchar *d_rightMean, int height, int width, int win1, int win2)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = x + y*width;
	uint32_t leftRes = 0;
	uint32_t rightRes = 0;
	int count = 0;

	for (int h = -win1; h <= win1; h++)
	{
		for (int w = -win2; w <= win2; w++)
		{
			if (h == 0 && w == 0) continue;
			uchar leftTemp = tex2D(leftTex, x + w, y + h);
			uchar rightTemp = tex2D(rightTex, x + w, y + h);
			if (d_leftMean[offset] > leftTemp)
			{
				leftRes = leftRes | (1 << count);
			}
			if (d_rightMean[offset] > rightTemp)
			{
				rightRes = rightRes | (1 << count);
			}
			count++;
			if (count == 32)
			{
				d_leftCen[y*width * 2 + 2 * x] = leftRes;
				d_rightCen[y*width * 2 + 2 * x] = rightRes;
				leftRes = 0;
				rightRes = 0;
				count = 0;
			}
		}
	}
	d_leftCen[y*width * 2 + 2 * x + 1] = leftRes;
	d_rightCen[y*width * 2 + 2 * x + 1] = rightRes;
}



__device__ int hamming_distance(uint64_t c1, uint64_t c2)
{
	return __popcll(c1^c2);
}

__device__ int hamming_distance(uint32_t c1, uint32_t c2)
{
	return __popcll(c1^c2);
}


__device__ void search_best_disp(uint32_t* leftCen, uint32_t* rightCen, float* leftPhi, float* rightPhi,
	int width, int height, int x, int y, int minDisp, int maxDisp, int &bestDx, int &bestScore)
{
	for (int dx = minDisp; dx <= maxDisp; dx++)
	{
		int cxR = x - dx;
		if (cxR < 0) continue;
		if (leftPhi)
		{
			float phiT = 0.25f;
			float phiL = leftPhi[y*width + x];
			float phiR = rightPhi[y*width + cxR];
			float dPhi = abs(phiL - phiR);
			if (dPhi > phiT) continue;
			if (phiL < 0) continue;
			if (phiR < 0) continue;
		}
		uint32_t c1 = leftCen[y*width * 2 + 2 * x];
		uint32_t c2 = rightCen[y*width * 2 + 2 * cxR];
		int d1 = hamming_distance(c1, c2);
		c1 = leftCen[y*width * 2 + 2 * x + 1];
		c2 = rightCen[y*width * 2 + 2 * cxR + 1];
		int d2 = hamming_distance(c1, c2);
		int d = d1 + d2;
		if (d < bestScore)
		{
			bestScore = d;
			bestDx = dx;
		}
	}
}


__global__ void disp_image_init(int* d_dispImg, int* d_scoreImg, float* d_leftPhi, float* d_rightPhi,
	uint32_t* d_leftCen, uint32_t* d_rightCen, int height, int width, int minDisp, int maxDisp,
	int dispRange, int randTimes, curandState *d_states)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = x + y*width;

	d_scoreImg[offset] = 1000;
	d_dispImg[offset] = minDisp;

	int bestScore = 1000;
	int bestDisp = minDisp;
	for (int t = 0; t < randTimes; t++)
	{
		int d = curand(d_states + offset) % dispRange + minDisp;

		//if (x == 128 && y == 128) printf("%d ", d);
		int minD = d;
		int maxD = d;
		int score = 1000;

		search_best_disp(d_leftCen, d_rightCen, d_leftPhi, d_rightPhi, width, height, x, y, minD, maxD, d, score);

		if (score < bestScore)
		{
			bestScore = score;
			bestDisp = d;
		}
	}
	d_scoreImg[offset] = bestScore;
	d_dispImg[offset] = bestDisp;
}

__global__ void left_to_right(int* d_dispImg, int* d_scoreImg, uint32_t* d_leftCen, uint32_t* d_rightCen,
	float* d_leftPhi, float* d_rightPhi, int height, int width, int minDisp, int maxDisp)
{
	int y = blockIdx.x*blockDim.x + threadIdx.x;
	for (int x = 1; x < width; x++)
	{
		int offset = y*width + x;
		int x0 = x - 1, y0 = y;
		int d0 = d_dispImg[y0*width + x0];
		if (d0 <= minDisp) continue;

		int score = d_scoreImg[offset];
		int minD = d0 - 1;
		int maxD = d0 + 1;
		int bestD = minDisp;
		int bestScore = 1000;
		search_best_disp(d_leftCen, d_rightCen, d_leftPhi, d_rightPhi, width, height, x, y, minD, maxD, bestD, bestScore);
		if (bestScore < score)
		{
			if (bestD < maxDisp)
			{
				d_dispImg[offset] = bestD;
				d_scoreImg[offset] = bestScore;
			}
		}
	}
}


__global__ void right_to_left(int* d_dispImg, int* d_scoreImg, uint32_t* d_leftCen, uint32_t* d_rightCen,
	float* d_leftPhi, float* d_rightPhi, int height, int width, int minDisp, int maxDisp)
{
	int y = blockIdx.x*blockDim.x + threadIdx.x;
	for (int x = width - 1 - 1; x >= 0; x--)
	{
		int offset = y*width + x;
		int x0 = x + 1, y0 = y;
		int d0 = d_dispImg[y0*width + x0];
		if (d0 <= minDisp) continue;
		int score = d_scoreImg[offset];
		int minD = d0 - 1;
		int maxD = d0 + 1;
		int bestD = minDisp;
		int bestScore = 1000;

		search_best_disp(d_leftCen, d_rightCen, d_leftPhi, d_rightPhi, width, height, x, y, minD, maxD, bestD, bestScore);
		if (bestScore < score)
		{
			if (bestD < maxDisp)
			{
				d_dispImg[offset] = bestD;
				d_scoreImg[offset] = bestScore;
			}
		}
	}
}


__global__ void up_to_down(int* d_dispImg, int* d_scoreImg, uint32_t* d_leftCen, uint32_t* d_rightCen,
	float* d_leftPhi, float* d_rightPhi, int height, int width, int minDisp, int maxDisp)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	for (int y = 1; y < height; y++)
	{
		int offset = y*width + x;
		int x0 = x, y0 = y - 1;
		int d0 = d_dispImg[y0*width + x0];
		if (d0 <= minDisp) continue;
		int score = d_scoreImg[offset];
		int minD = d0 - 1;
		int maxD = d0 + 1;
		int bestD = minDisp;
		int bestScore = 1000;
		search_best_disp(d_leftCen, d_rightCen, d_leftPhi, d_rightPhi, width, height, x, y, minD, maxD, bestD, bestScore);
		if (bestScore < score)
		{
			if (bestD < maxDisp)
			{
				d_dispImg[offset] = bestD;
				d_scoreImg[offset] = bestScore;
			}
		}
	}
}


__global__ void down_to_up(int* d_dispImg, int* d_scoreImg, uint32_t* d_leftCen, uint32_t* d_rightCen,
	float* d_leftPhi, float* d_rightPhi, int height, int width, int minDisp, int maxDisp)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	for (int y = height - 1 - 1; y >= 0; y--)
	{
		int offset = y*width + x;
		int x0 = x, y0 = y + 1;
		int d0 = d_dispImg[y0*width + x0];
		if (d0 <= minDisp) continue;
		int score = d_scoreImg[offset];
		int minD = d0 - 1;
		int maxD = d0 + 1;
		int bestD = minDisp;
		int bestScore = 1000;
		search_best_disp(d_leftCen, d_rightCen, d_leftPhi, d_rightPhi, width, height, x, y, minD, maxD, bestD, bestScore);
		if (bestScore < score)
		{
			if (bestD < maxDisp)
			{
				d_dispImg[offset] = bestD;
				d_scoreImg[offset] = bestScore;
			}
		}
	}
}


__global__ void median_filter(int *d_src, int *d_dst, int height, int width)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int row = idx / width;
	int col = idx % width;
	const int n = 3;
	int win[n*n];
	int half = n / 2;
	if (row >= half && col >= half && row < height - half && col < width - half)
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				win[i*n + j] = d_src[(row - half + i)*width + col - half + j];
			}
		}
		for (int i = 0; i < (n*n) / 2 + 1; i++)
		{
			int minIdx = i;
			for (int j = i + 1; j < n*n; j++)
			{
				if (win[j] < win[minIdx])
				{
					minIdx = j;
				}
			}
			const int temp = win[i];
			win[i] = win[minIdx];
			win[minIdx] = temp;
		}
		d_dst[idx] = win[(n*n) / 2];
	}
	else
	{
		d_dst[idx] = d_src[idx];
	}
}

__global__ void median_filter2(int *d_src, int *d_dst, int height, int width)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = y*width + x;
	const int winSize = 3;
	const int halfSize = winSize / 2;
	int win[winSize*winSize];

	// first step: horizontal median filter
	if (x >= winSize && y >= winSize && x < width - winSize && y < height - winSize)
	{
		int i = 0;
		int j = 0;
		int temp = 0;
		for (int x2 = x - winSize; x2 <= x + winSize; x2++)
		{
			temp = d_src[y*width + x2];
			i = j - 1;
			while (i >= 0 && win[i] > temp)
			{
				win[i + 1] = win[i];
				i--;
			}
			win[i + 1] = temp;
			j++;
		}
		d_dst[y*width + x] = win[winSize];
	}
	else
	{
		d_dst[y*width + x] = d_src[y*width + x];
	}
	__syncthreads();
	// second step: vertical median filter
	if (x >= winSize && y >= winSize && x < width - winSize && y < height - winSize)
	{
		int i = 0;
		int j = 0;
		int temp = 0;
		for (int y2 = y - winSize; y2 <= y + winSize; y2++)
		{
			temp = d_dst[y2*width + x];
			i = j - 1;
			while (i >= 0 && win[i] > temp)
			{
				win[i + 1] = win[i];
				i--;
			}
			win[i + 1] = temp;
			j++;
		}
		d_dst[y*width + x] = win[winSize];
	}
	else
	{
		d_dst[y*width + x] = d_src[y*width + x];
	}
}


__device__ int FindRoot(int *d_labelImg, int label)
{
	while (d_labelImg[label] != label)
	{
		label = d_labelImg[label];
	}
	return label;
}

__device__ void Union(int *d_dispImg, int *d_labelImg, int address0, int address1, int *sChanged)
{
	if (fabsf(d_dispImg[address0] - d_dispImg[address1]) <= 2)
	{
		int root0 = FindRoot(d_labelImg, address0);
		int root1 = FindRoot(d_labelImg, address1);

		if (root0 < root1)
		{
			atomicMin(d_labelImg + root1, root0);
			sChanged[0] = 1;
		}
		else if (root1 < root0)
		{
			atomicMin(d_labelImg + root0, root1);
			sChanged[0] = 1;
		}
	}
}


__global__ void block_label(int *d_dispImg, int *d_labelImg, int height, int width)
{
	__shared__ int sSegs[BLOCK_SIZE*BLOCK_SIZE];
	__shared__ int sLabels[BLOCK_SIZE*BLOCK_SIZE];

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = y*width + x;
	int l_x = x % BLOCK_SIZE;
	int l_y = y % BLOCK_SIZE;
	int l_idx = l_y*BLOCK_SIZE + l_x;
	sSegs[l_idx] = d_dispImg[idx];

	__shared__ int sChanged[1];

	__syncthreads();

	int label = l_idx;
	//int n_l_x[8], n_l_y[8];
	//n_l_x[0] = l_x - 1;  n_l_y[0] = l_y - 1;
	//n_l_x[1] = l_x;      n_l_y[1] = l_y - 1;
	//n_l_x[2] = l_x + 1;  n_l_y[2] = l_y - 1;
	//n_l_x[3] = l_x - 1;  n_l_y[3] = l_y;
	//n_l_x[4] = l_x + 1;  n_l_y[4] = l_y;
	//n_l_x[5] = l_x - 1;  n_l_y[5] = l_y + 1;
	//n_l_x[6] = l_x;      n_l_y[6] = l_y + 1;
	//n_l_x[7] = l_x + 1;  n_l_y[7] = l_y + 1;
	const int neighArea = 4;
	int n_l_x[neighArea], n_l_y[neighArea];
	n_l_x[0] = l_x - 1;    n_l_y[0] = l_y;
	n_l_x[1] = l_x + 1;    n_l_y[1] = l_y;
	n_l_x[2] = l_x;        n_l_y[2] = l_y - 1;
	n_l_x[3] = l_x;        n_l_y[3] = l_y + 1;

	while (1)
	{
		sLabels[l_idx] = label;
		if (threadIdx.x == 0 && threadIdx.y == 0) sChanged[0] = 0;
		__syncthreads();

		int newLabel = label;
		for (int i = 0; i < neighArea; i++)
		{
			if (n_l_x[i] >= 0 && n_l_x[i] < BLOCK_SIZE && n_l_y[i] >= 0 && n_l_y[i] < BLOCK_SIZE)
			{
				int n_l_idx = n_l_y[i] * BLOCK_SIZE + n_l_x[i];
				/*if (sSegs[l_idx] == 255 && sSegs[n_l_idx] == 255)
				{
				newLabel = static_cast<int>(fminf(newLabel, sLabels[n_l_idx]));
				}*/
				if (fabsf(sSegs[l_idx] - sSegs[n_l_idx]) <= 1)
				{
					newLabel = static_cast<int>(fminf(newLabel, sLabels[n_l_idx]));
				}
			}
		}
		__syncthreads();

		if (newLabel < label)
		{
			atomicMin(sLabels + label, newLabel);
			sChanged[0] = 1;
		}
		__syncthreads();

		if (sChanged[0] == 0) break;
		label = FindRoot(sLabels, label);
		__syncthreads();

	}
	/*if (d_dispImg[idx] == 0)
	{
	d_labelImg[idx] = 0;
	}
	else
	{
	d_labelImg[idx] = (blockIdx.y*blockDim.y + label / BLOCK_SIZE)*width + blockIdx.x*blockDim.x + label % BLOCK_SIZE;
	}*/
	d_labelImg[idx] = (blockIdx.y*blockDim.y + label / BLOCK_SIZE)*width + blockIdx.x*blockDim.x + label % BLOCK_SIZE;
	//d_labelImg[idx] = label;
}

__global__ void block_merge(int *d_dispImg, int *d_labelImg, int height, int width)
{
	dim3 subBlockIdx(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y);

	dim3 subBlockDim(BLOCK_SIZE, BLOCK_SIZE);
	int rep = subBlockDim.x / blockDim.z;

	__shared__ int sChanged[1];

	while (1)
	{
		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
		{
			sChanged[0] = 0;
		}
		__syncthreads();

		for (int i = 0; i < rep; i++)
		{
			int x = subBlockIdx.x*subBlockDim.x + i*blockDim.z + threadIdx.z;
			int y = (subBlockIdx.y + 1)*subBlockDim.y - 1;
			if (y + 1 < height)
			{
				int address0 = y*width + x;
				int address1 = (y + 1)*width + x;
				Union(d_dispImg, d_labelImg, address0, address1, sChanged);
			}
		}

		for (int i = 0; i < rep; i++)
		{
			int x = (subBlockIdx.x + 1)*subBlockDim.x - 1;
			int y = subBlockIdx.y*subBlockDim.y + i*blockDim.z + threadIdx.z;
			if (x + 1 < width)
			{
				int address0 = y*width + x;
				int address1 = y*width + x + 1;
				Union(d_dispImg, d_labelImg, address0, address1, sChanged);
			}
		}
		__syncthreads();

		if (sChanged[0] == 0) break;

		__syncthreads();
	}
}


__global__ void calcu_area(int *d_dispImg, int *d_labelImg, int *d_areaImg, int height, int width)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = y*BLOCK_SIZE*width + x;

	int currLabel = FindRoot(d_labelImg, idx);
	int nextLabel;
	int count = 1;

	for (int i = 1; i < BLOCK_SIZE; i++)
	{
		idx = (y*BLOCK_SIZE + i)*width + x;
		nextLabel = FindRoot(d_labelImg, idx);
		if (currLabel != nextLabel)
		{
			atomicAdd(d_areaImg + currLabel, count);
			currLabel = nextLabel;
			count = 1;
		}
		else
		{
			count++;
		}
		if (i == BLOCK_SIZE - 1)
		{
			atomicAdd(d_areaImg + currLabel, count);
		}
	}
	__syncthreads();
}


__global__ void remove_small_segments(int *d_dispImg, int *d_labelImg, int *d_areaImg, int height, int width, int speckleSize, int minDisp)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = y*width + x;
	int label = FindRoot(d_labelImg, idx);
	if (d_areaImg[label] < speckleSize)
	{
		d_dispImg[idx] = minDisp;
	}
}

// src0---half black / half white
// src1---half white / half black
int SegBW2(IplImage* src0, IplImage* bina)
{
	if (src0 == NULL)
		return -1;

	int w, h, ws8;
	w = src0->width;
	h = src0->height;
	ws8 = src0->widthStep;

	float T = 0;
	for (int r = 0; r < h; r++)
	{
		for (int c = 0; c < w; c++)
		{
			unsigned char* pRow = (unsigned char*)src0->imageData + r*ws8;
			T = T + float(pRow[c]);
		}
	}
	T /= w*h;

	for (int r = 0; r < h; r++)
	{
		unsigned char* pRow1 = (unsigned char*)src0->imageData + r*ws8;
		unsigned char* pRowB = (unsigned char*)bina->imageData + r*ws8;
		for (int c = 0; c < w; c++)
		{
			//int diff = pRow1[c] - pRow2[c];
			if (pRow1[c]>int(T))
			{
				pRowB[c] = 255;
			}
			else
			{
				pRowB[c] = 0;
			}
		}
	}

	return 0;
}

// unwrapping for a planar target using a binary pattern
// nFringe----the number of fringes
int UnwrappingPhase(IplImage* bina, float *rtmPhi, float *absPhi, int nFringes)
{
	int w = bina->width;
	int h = bina->height;

	int w2 = w / 4;

	float pi = 3.1415926f;

	for (int r = 0; r<h; r++)
	{
		unsigned char* binaRow = (unsigned char*)bina->imageData + r*w;
		float *rtmPhiRow = rtmPhi + r*w;
		float *absPhiRow = absPhi + r*w;

		for (int c = w2; c < w - w2; c++)
		{
			if (rtmPhiRow[c] <= -4)
			{
				continue;
			}
			if (binaRow[c] - binaRow[c - 1] > 0)
			{
				r = r;
				int k = 0;

				k = nFringes / 2;
				for (int c2 = c; c2<w; c2++)
				{
					if (rtmPhiRow[c2] <= -4)
					{
						continue;
					}
					if ((rtmPhiRow[c2] - rtmPhiRow[c2 - 1])<-pi)
					{
						k += 1;
					}
					absPhiRow[c2] = k * 2 * pi + rtmPhiRow[c2];
				}

				k = nFringes / 2;
				for (int c2 = c; c2 >= 0; c2--)
				{
					if (rtmPhiRow[c2] <= -4)
					{
						continue;
					}
					if ((rtmPhiRow[c2] - rtmPhiRow[c2 + 1])>pi)
					{
						k -= 1;
					}
					absPhiRow[c2] = k * 2 * pi + rtmPhiRow[c2];

					//kMap[r*w + c2] = k;
				}

				//kMap[r*w + c] = 1;
				break;
			}
		}
	}

	//OutData2Txt(kMap, w, h, w, "d:/kMap.txt");

	//delete[] kMap;

	return 0;
}

//compute the absolute phase map for the reference image
void RefImgAbsPhase(IplImage* binaImages, int nBinaImages, float *absPhi, float* retPhiRef, float nFringes)
{
	float pi = 3.1415926f;

	int w = binaImages->width;
	int h = binaImages->height;

	//float* rtmPhi =new float[w*h];
	float *column = new float[w*h];
	//memset(retPhiRef, 0, sizeof(float)*w*h);

	int diffT = 18;
	IplImage* bina = cvCreateImage(cvSize(w, h), 8, 1);
	//SegBW(binaImages[0], binaImages[1], bina);
	//cvSaveImage("d:/bw.bmp", bina);
	SegBW2(binaImages, bina);
	cvSaveImage("bw2.bmp", bina);

	//WrapPhaseShift(fringeImages, nFringeImages, retPhiRef, diffT);


	UnwrappingPhase(bina, retPhiRef, absPhi, nFringes);

	//OutData2Txt(retPhiRef, w, h, w, "rtmPhiRef.txt");
	//OutData2Txt(absPhi, w, h, w, "absPhiRef.txt");

	delete[] column;
	cvReleaseImage(&bina);

	return;
}

// unwrapping using the reference image
int Unwrapping_RefImg(int w, int h, float *rtmPhi, float *h_rightPhi, float *refPhi, float *absPhi, int *disp, int minDisp)
{
	float *phi0 = new float[w*h];
	float *rphi0 = new float[w*h];

	float pi = 3.14159f;

	for (int r = 0; r < h; r++)
	{
		for (int c = 0; c<w; c++)
		{
			if ((r == 179) && (c == 108))
			{
				r = r;
			}
			int idx = r*w + c;
			if (rtmPhi[r*w + c] <= -4)
				continue;

			int dx = int(disp[idx] - 0.5f);

			if (dx <= minDisp)
			{
				//absPhi[idx] = 0;
				continue;
			}
			int xRef = c - dx;
			if (xRef < 0)
				continue;
			float roughPhi = refPhi[r*w + xRef];

			phi0[idx] = roughPhi;

			rphi0[idx] = h_rightPhi[r*w + xRef];

			int k = int((roughPhi - rtmPhi[idx]) / (2 * pi) + 0.5f);

			absPhi[idx] = 2 * k*pi + rtmPhi[idx];
		}
	}

	//OutData2Txt(phi0, w, h, w, "phi0.txt");
	//OutData2Txt(rphi0, w, h, w, "rphi0.txt");

	delete[] phi0;
	delete[] rphi0;

	return 0;
}



int main(int argc, char* argv[])
{
	IplImage* leftImg = cvLoadImage("tsetImage\\david\\speckle.bmp", 0);
	IplImage* rightImg = cvLoadImage("tsetImage\\ref\\ref.bmp", 0);

	IplImage* objFringeImg[3];
	IplImage* refFringeImg[3];
	objFringeImg[0] = cvLoadImage("tsetImage\\david\\objFringe0.bmp", 0);
	objFringeImg[1] = cvLoadImage("tsetImage\\david\\objFringe1.bmp", 0);
	objFringeImg[2] = cvLoadImage("tsetImage\\david\\objFringe2.bmp", 0);
	refFringeImg[0] = cvLoadImage("tsetImage\\ref\\refFringe0.bmp", 0);
	refFringeImg[1] = cvLoadImage("tsetImage\\ref\\refFringe1.bmp", 0);
	refFringeImg[2] = cvLoadImage("tsetImage\\ref\\refFringe2.bmp", 0);

	IplImage* binaImg = cvLoadImage("tsetImage\\cam_00.bmp", 0);

	const unsigned int height = 480;
	const unsigned int width = 640;
	const unsigned int imgSize = height*width;


	uchar* d_objFriImg, *d_refFriImg;
	if (cudaSuccess != cudaMalloc((void **)&d_objFriImg, 3 * imgSize * sizeof(uchar)))
		std::cout << "device malloc object fringe image error" << std::endl;
	if (cudaSuccess != cudaMalloc((void **)&d_refFriImg, 3 * imgSize * sizeof(uchar)))
		std::cout << "device malloc reference fringe image error" << std::endl;

	float* d_leftPhi, *d_rightPhi;
	if (cudaSuccess != cudaMalloc((void **)&d_leftPhi, imgSize * sizeof(float)))
		std::cout << "device malloc left phase image error" << std::endl;
	if (cudaSuccess != cudaMemset(d_leftPhi, 0.0f, imgSize * sizeof(float)))
		std::cout << "device memset left phase image error" << std::endl;
	if (cudaSuccess != cudaMalloc((void **)&d_rightPhi, imgSize * sizeof(float)))
		std::cout << "device malloc right phase image error" << std::endl;
	if (cudaSuccess != cudaMemset(d_rightPhi, 0.0f, imgSize * sizeof(float)))
		std::cout << "device memset right phase image error" << std::endl;


	if (cudaSuccess != cudaMemcpy(d_objFriImg, objFringeImg[0]->imageData, imgSize * sizeof(uchar), cudaMemcpyHostToDevice))
		std::cout << "copy object fringe image 0 from host to devcie error" << std::endl;
	if (cudaSuccess != cudaMemcpy(d_objFriImg + imgSize, objFringeImg[1]->imageData, imgSize * sizeof(uchar), cudaMemcpyHostToDevice))
		std::cout << "copy object fringe image 1 from host to device error" << std::endl;
	if (cudaSuccess != cudaMemcpy(d_objFriImg + 2 * imgSize, objFringeImg[2]->imageData, imgSize * sizeof(uchar), cudaMemcpyHostToDevice))
		std::cout << "copy object fringe image 2 from host to device error" << std::endl;
	if (cudaSuccess != cudaMemcpy(d_refFriImg, refFringeImg[0]->imageData, imgSize * sizeof(uchar), cudaMemcpyHostToDevice))
		std::cout << "copy reference fringe image 0 from host to devcie error" << std::endl;
	if (cudaSuccess != cudaMemcpy(d_refFriImg + imgSize, refFringeImg[1]->imageData, imgSize * sizeof(uchar), cudaMemcpyHostToDevice))
		std::cout << "copy reference fringe image 1 from host to device error" << std::endl;
	if (cudaSuccess != cudaMemcpy(d_refFriImg + 2 * imgSize, refFringeImg[2]->imageData, imgSize * sizeof(uchar), cudaMemcpyHostToDevice))
		std::cout << "copy reference fringe image 2 from host to devcie error" << std::endl;


	uchar* d_leftImg, *d_rightImg;
	if (cudaSuccess != cudaMalloc((void **)&d_leftImg, imgSize * sizeof(uchar)))
		std::cout << "device malloc left image error" << std::endl;
	if (cudaSuccess != cudaMalloc((void **)&d_rightImg, imgSize * sizeof(uchar)))
		std::cout << "device malloc right image error" << std::endl;

	if (cudaSuccess != cudaMemcpy(d_leftImg, leftImg->imageData, imgSize * sizeof(uchar), cudaMemcpyHostToDevice))
		std::cout << "copy left image from host to device error" << std::endl;
	if (cudaSuccess != cudaMemcpy(d_rightImg, rightImg->imageData, imgSize * sizeof(uchar), cudaMemcpyHostToDevice))
		std::cout << "copy right image from host to device error" << std::endl;

	uint32_t* d_leftCen, *d_rightCen;
	if (cudaSuccess != cudaMalloc((void **)&d_leftCen, 2 * imgSize * sizeof(uint32_t)))
		std::cout << "device malloc left census error" << std::endl;
	if (cudaSuccess != cudaMalloc((void **)&d_rightCen, 2 * imgSize * sizeof(uint32_t)))
		std::cout << "device malloc right census error" << std::endl;

	uchar* d_leftMean, *d_rightMean;
	if (cudaSuccess != cudaMalloc((void **)&d_leftMean, imgSize * sizeof(uchar)))
		std::cout << "device malloc left mean error" << std::endl;
	if (cudaSuccess != cudaMalloc((void **)&d_rightMean, imgSize * sizeof(uchar)))
		std::cout << "device malloc right mean error" << std::endl;

	int* d_dispImg, *d_scoreImg;
	if (cudaSuccess != cudaMalloc((void **)&d_dispImg, imgSize * sizeof(int)))
		std::cout << "device malloc disparity image error" << std::endl;
	if (cudaSuccess != cudaMalloc((void **)&d_scoreImg, imgSize * sizeof(int)))
		std::cout << "device malloc score image error" << std::endl;

	int *d_postImg;
	if (cudaSuccess != cudaMalloc((void **)&d_postImg, imgSize * sizeof(int)))
		std::cout << "device malloc post-processing image error" << std::endl;

	int *d_labelImg;
	if (cudaSuccess != cudaMalloc((void **)&d_labelImg, imgSize * sizeof(int)))
		std::cout << "device malloc label image error" << std::endl;

	int *d_areaImg;
	if (cudaSuccess != cudaMalloc((void **)&d_areaImg, imgSize * sizeof(int)))
		std::cout << "device malloc area image error" << std::endl;

	leftTex.addressMode[0] = cudaAddressModeClamp;
	leftTex.addressMode[1] = cudaAddressModeClamp;
	rightTex.addressMode[0] = cudaAddressModeClamp;
	rightTex.addressMode[1] = cudaAddressModeClamp;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar>();

	if (cudaSuccess != cudaBindTexture2D(NULL, &leftTex, d_leftImg, &desc, width, height, width * sizeof(uchar)))
		std::cout << "bind left texture error" << std::endl;
	if (cudaSuccess != cudaBindTexture2D(NULL, &rightTex, d_rightImg, &desc, width, height, width * sizeof(uchar)))
		std::cout << "bind right texture error" << std::endl;

	curandState *d_states;
	cudaMalloc((void **)&d_states, height*width * sizeof(curandState));

	int *d_doneImg;
	cudaMalloc((void **)&d_doneImg, height*width * sizeof(int));
	cudaMemset(d_doneImg, 0, imgSize * sizeof(int));
	int *segList_x, *segList_y;
	cudaMalloc((void **)&segList_x, imgSize * sizeof(int));
	cudaMalloc((void **)&segList_y, imgSize * sizeof(int));



	int minDisp = -50;
	int maxDisp = 165;
	int dispRange = maxDisp - minDisp;
	int randTimes = 3;

	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	dim3 blockSize1(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize1(width / blockSize1.x, height / blockSize1.y);

	dim3 blockSize2(4, 4, BLOCK_SIZE);
	dim3 gridSize2(width / (2 * BLOCK_SIZE), height / (2 * BLOCK_SIZE));

	dim3 blockSize3(BLOCK_SIZE, 1);
	dim3 gridSize3(width / BLOCK_SIZE, height / BLOCK_SIZE);

	dim3 blockSize4(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize4(width / BLOCK_SIZE, height / BLOCK_SIZE);

	rand_init << <gridSize, blockSize >> > (d_states, height, width);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, NULL);

	int speckleSize = 100;

	int diffT = 18;
	wrap_phase_shift << <gridSize, blockSize >> > (d_objFriImg, d_leftPhi, height, width, diffT);
	wrap_phase_shift << <gridSize, blockSize >> > (d_refFriImg, d_rightPhi, height, width, diffT);

	int win1 = 3, win2 = 4;

	mean_filter << <gridSize, blockSize >> > (d_leftMean, d_rightMean, d_leftImg, d_rightImg, height, width, win1, win2);

	census_transform32 << <gridSize, blockSize >> > (d_leftCen, d_rightCen, d_leftMean, d_rightMean, height, width, win1, win2);

	//uint32_t *h_leftCen = new uint32_t[2 * imgSize]();
	//uint32_t *h_rightCen = new uint32_t[2 * imgSize]();
	//cudaMemcpy(h_leftCen, d_leftCen, 2 * imgSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_rightCen, d_rightCen, 2 * imgSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	//float *h_leftPhi = new float[imgSize]();
	//float *h_rightPhi = new float[imgSize]();
	//cudaMemcpy(h_leftPhi, d_leftPhi, imgSize * sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_rightPhi, d_rightPhi, imgSize * sizeof(float), cudaMemcpyDeviceToHost);
	//uchar *h_leftMean = new uchar[imgSize]();
	//uchar *h_rightMean = new uchar[imgSize]();
	//cudaMemcpy(h_leftMean, d_leftMean, imgSize * sizeof(uchar), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_rightMean, d_rightMean, imgSize * sizeof(uchar), cudaMemcpyDeviceToHost);



	disp_image_init << <gridSize, blockSize >> > (d_dispImg, d_scoreImg, d_leftPhi, d_rightPhi, d_leftCen, d_rightCen, height, width, minDisp, maxDisp, dispRange, randTimes, d_states);

	left_to_right << <height / WARP_SIZE, WARP_SIZE >> > (d_dispImg, d_scoreImg, d_leftCen, d_rightCen, d_leftPhi, d_rightPhi, height, width, minDisp, maxDisp);
	up_to_down << <width / WARP_SIZE, WARP_SIZE >> > (d_dispImg, d_scoreImg, d_leftCen, d_rightCen, d_leftPhi, d_rightPhi, height, width, minDisp, maxDisp);
	right_to_left << <height / WARP_SIZE, WARP_SIZE >> > (d_dispImg, d_scoreImg, d_leftCen, d_rightCen, d_leftPhi, d_rightPhi, height, width, minDisp, maxDisp);
	down_to_up << <width / WARP_SIZE, WARP_SIZE >> > (d_dispImg, d_scoreImg, d_leftCen, d_rightCen, d_leftPhi, d_rightPhi, height, width, minDisp, maxDisp);

	median_filter2 << <gridSize, blockSize >> > (d_dispImg, d_postImg, height, width);



	block_label << <gridSize1, blockSize1 >> > (d_postImg, d_labelImg, height, width);
	block_merge << <gridSize2, blockSize2 >> > (d_postImg, d_labelImg, height, width);
	calcu_area << <gridSize3, blockSize3 >> > (d_postImg, d_labelImg, d_areaImg, height, width);
	remove_small_segments << <gridSize4, blockSize4 >> > (d_postImg, d_labelImg, d_areaImg, height, width, speckleSize, minDisp);

	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);
	printf("GPU processing time : %.4f (ms)\n", msecTotal);

	float *h_leftPhi = new float[imgSize]();
	float *h_rightPhi = new float[imgSize]();
	float *h_absPhiRef = new float[imgSize]();
	int* h_dispImg = new int[height*width]();
	if (cudaMemcpy(h_leftPhi, d_leftPhi, imgSize * sizeof(float), cudaMemcpyDeviceToHost))
		std::cout << "copy left phase  image from device to host error" << std::endl;
	if (cudaMemcpy(h_rightPhi, d_rightPhi, imgSize * sizeof(float), cudaMemcpyDeviceToHost))
		std::cout << "copy right phase image from device to host error" << std::endl;
	if (cudaMemcpy(h_dispImg, d_postImg, height*width * sizeof(int), cudaMemcpyDeviceToHost))
		std::cout << "copy disparity image from device to host error" << std::endl;

	int nBinaImg = 1;
	float n_finges = 25.6f;
	RefImgAbsPhase(binaImg, nBinaImg, h_absPhiRef, h_rightPhi, n_finges);

	float *h_absPhi = new float[imgSize]();
	Unwrapping_RefImg(width, height, h_leftPhi, h_rightPhi, h_absPhiRef, h_absPhi, h_dispImg, minDisp);

	cv::Mat h_dispMat(height, width, CV_8UC1, cv::Scalar(0));
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			h_dispMat.at<uchar>(row, col) = static_cast<uchar>(h_dispImg[row*width + col]);
			//std::cout << h_dispImg[row*width + col] << " ";
		}
		//std::cout << std::endl;
	}

	cv::Mat absPhiMat(height, width, CV_32FC1, cv::Scalar(0));
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			absPhiMat.at<float>(row, col) = h_absPhi[row*width + col];
		}
	}
	cv::normalize(absPhiMat, absPhiMat, 255, 0, cv::NORM_MINMAX);
	absPhiMat.convertTo(absPhiMat, CV_8U);
	IplImage absPhiImage(absPhiMat);
	cvShowImage("absPhi", &absPhiImage);


	IplImage dispShowImg(h_dispMat);
	//IplImage leftPhiImg(leftPhiMat);
	//IplImage leftMeanImg(leftMeanMat);
	cvShowImage("dispImg", &dispShowImg);
	cvSaveImage("dispImg.bmp", &dispShowImg);
	//cvShowImage("leftPhiImg", &leftPhiImg);
	//cvShowImage("leftMeanImg", &leftMeanImg);
	cvWaitKey(0);
}