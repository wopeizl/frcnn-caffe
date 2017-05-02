// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------
#include <cub/cub.cuh>
#include <iomanip>
#include <cfloat>
#include <cmath>

#include "caffe/FRCNN/frcnn_proposal_layer.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"  
#include "caffe/FRCNN/util/frcnn_gpu_nms.hpp"  
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

#include <sys/time.h>
static double getTimeOfMSeconds() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec*1000. + tv.tv_usec/1000.;
}

static inline
__host__ __device__ int DIVUP(const int m, const int n) {
	return (m+n-1) / (n);
}
namespace caffe {

namespace Frcnn {

using std::vector;

__global__ void EnumerateShiftedAnchors(
	const int height, const int width, const int num_anchors,
	const float4* anchors, int4* shifted_anchors, const int feat_stride) {
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	if (tidx < width && tidy < height) {
		const int ret_x = tidx*feat_stride;
		const int ret_y = tidy*feat_stride;
		for (int k = 0; k < num_anchors; k++) {
			int4 ret;
			ret.x = anchors[k].x + ret_x;
			ret.y = anchors[k].y + ret_y;
			ret.z = anchors[k].z + ret_x;
			ret.w = anchors[k].w + ret_y;
			int index = (tidy*width+tidx)*num_anchors + k;
			shifted_anchors[index] = ret;
		}
	}
}
__global__ void EnumerateProposal(
	const int height, const int width, const int num_anchors, 
	const int im_min_size, const int im_height, const int im_width,
	const int4* shifted_anchors, const float* bbox,
	const float* scores, float4* proposal, float* scores_kept) {
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	if (tidx < width && tidy < height) {
		for (int k = 0; k < num_anchors; ++k) {
			float4 ret_deltas;
			int index = 4*k*height*width + tidy*width + tidx;
			ret_deltas.x = bbox[index];
			ret_deltas.y = bbox[index+height*width];
			ret_deltas.z = bbox[index+2*height*width];
			ret_deltas.w = bbox[index+3*height*width];
			index = tidy*width + tidx + (num_anchors+k)*(height*width);
			float score = scores[index];
			index = (tidy*width+tidx)*num_anchors + k;
			const int4 ret_shifts_anchors = shifted_anchors[index];
			
			float ctr_x, ctr_y, w, h;
			float pred_ctr_x, pred_ctr_y, pred_w, pred_h;
			w = ret_shifts_anchors.z - ret_shifts_anchors.x + 1.0;
			h = ret_shifts_anchors.w - ret_shifts_anchors.y + 1.0;
			ctr_x = ret_shifts_anchors.x + 0.5*w;
			ctr_y = ret_shifts_anchors.y + 0.5*h;

			pred_ctr_x = ret_deltas.x * w + ctr_x;
			pred_ctr_y = ret_deltas.y * h + ctr_y;
			pred_w = exp(ret_deltas.z) * w;
			pred_h = exp(ret_deltas.w) * h;

			float4 ret;
			ret.x = max(min(pred_ctr_x - 0.5*pred_w, im_width-1.0), 0.0);
			ret.y = max(min(pred_ctr_y - 0.5*pred_h, im_height-1.0), 0.0);
			ret.z = max(min(pred_ctr_x + 0.5*pred_w, im_width-1.0), 0.0);
			ret.w = max(min(pred_ctr_y + 0.5*pred_h, im_height-1.0), 0.0);
			if ( (ret.z-ret.x+1 >= im_min_size) && (ret.w-ret.y+1 >= im_min_size) ) {
				proposal[index] = ret;
				scores_kept[index] = score;
			} else {
				scores_kept[index] = 0.0f;  //  
			}
		}
	}
}

int const threadsPerBlock = sizeof(unsigned long long) * 8;
__device__ inline float devIoU(const float4 a, const float4 b) {
	float left = max(a.x, b.x), right = min(a.z, b.z);
	float top = max(a.y, b.y), bottom = min(a.w, b.w);
	float width = max(right-left + 1, 0.f), height = max(bottom - top + 1, 0.f);
	float interS = width*height;
	float Sa = (a.z - a.x + 1) * (a.w - a.y + 1);
	float Sb = (b.z - b.x + 1) * (b.w - b.y + 1);
	return interS / (Sa + Sb - interS);
}
__global__ void NMSKernel(const int n_boxes, const float nmx_overlap_thresh,
	const float4* dev_boxes, unsigned long long *dev_mask) {
	const int row_start = blockIdx.y;
	const int col_start = blockIdx.x;
	const int row_size = min(n_boxes - row_start*threadsPerBlock, threadsPerBlock);
	const int col_size = min(n_boxes - col_start*threadsPerBlock, threadsPerBlock);
	__shared__ float4 block_boxes[threadsPerBlock];
	if (threadIdx.x < col_size) {
		block_boxes[threadIdx.x].x = dev_boxes[threadsPerBlock*col_start + threadIdx.x].x;
		block_boxes[threadIdx.x].y = dev_boxes[threadsPerBlock*col_start + threadIdx.x].y;
		block_boxes[threadIdx.x].z = dev_boxes[threadsPerBlock*col_start + threadIdx.x].z;
		block_boxes[threadIdx.x].w = dev_boxes[threadsPerBlock*col_start + threadIdx.x].w;
	}
	__syncthreads();
	if (threadIdx.x < row_size) {
		const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
		const float4 *cur_box = dev_boxes + cur_box_idx;
		int i = 0;
		unsigned long long t = 0;
		int start = 0;
		if ( row_start == col_start) {
			start = threadIdx.x + 1;
		}
		for (i = start; i < col_size; ++i) {
			if (devIoU(cur_box[0], block_boxes[i]) > nmx_overlap_thresh) {
				t |= 1ULL << i;
			}
		}
		const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
		dev_mask[cur_box_idx*col_blocks + col_start] = t;
	}
}
void NMS_gpu(int* keep_out, int*num_out, 
	const float4* boxes_dev, int boxes_num, float nmx_overlap_thresh) {
	unsigned long long* mask_dev = NULL;
	const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
	const size_t nSize = sizeof(unsigned long long)*boxes_num*col_blocks;
	CUDA_CHECK( cudaMalloc((void**)&mask_dev, nSize) );
	dim3 blocks(DIVUP(boxes_num, threadsPerBlock), DIVUP(boxes_num, threadsPerBlock));
	dim3 threads(threadsPerBlock);
	NMSKernel<<<blocks, threads>>>(boxes_num,
			nmx_overlap_thresh, 
			boxes_dev, 
			mask_dev);

	std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
	CUDA_CHECK( cudaMemcpy(&mask_host[0], mask_dev, nSize, cudaMemcpyDeviceToHost) );
	std::vector<unsigned long long> remv(col_blocks);
	memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);
	int num_top_keep = 0;
	for (int i = 0; i < boxes_num; ++i) {
		int nblock = i / threadsPerBlock;
		int inblock = i % threadsPerBlock;
		if (!(remv[nblock] & ( 1ULL << inblock))) {
			keep_out[num_top_keep++] = i;
			unsigned long long *p = &mask_host[0] + i * col_blocks;
			for (int j = nblock; j < col_blocks; ++j) {
				remv[j] |= p[j];
			}
		}
	}
	*num_out = num_top_keep;
	CUDA_CHECK( cudaFree(mask_dev) );
	mask_host.clear();
	remv.clear();
}
template <typename Dtype>
__global__ void ProposalForward(const int num_out, 
	const int* keep_out, const float4* bbox, Dtype* rois) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < num_out) {
		float4 ret = bbox[keep_out[index]];
		rois[5*index + 1] = ret.x;
		rois[5*index + 2] = ret.y;
		rois[5*index + 3] = ret.z;
		rois[5*index + 4] = ret.w;
		rois[5*index + 0] = 0;
	}
}

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	//Forward_cpu(bottom, top);
	//return;
	//double st = getTimeOfMSeconds();
    int pre_nms_topN;
    int post_nms_topN;
    float min_size;
    float nms_thresh;

    if (this->phase_ == TRAIN) {
        pre_nms_topN = FrcnnParam::rpn_pre_nms_top_n;
        post_nms_topN = FrcnnParam::rpn_post_nms_top_n;
        nms_thresh = FrcnnParam::rpn_nms_thresh;
        min_size = FrcnnParam::rpn_min_size;
    }
    else {
        pre_nms_topN = FrcnnParam::test_rpn_pre_nms_top_n;
        post_nms_topN = FrcnnParam::test_rpn_post_nms_top_n;
        nms_thresh = FrcnnParam::test_rpn_nms_thresh;
        min_size = FrcnnParam::test_rpn_min_size;
    }

	const Dtype* bottom_scores = bottom[0]->gpu_data();
	const Dtype* bottom_bbox_deltas = bottom[1]->gpu_data();
	const Dtype* bottom_im_info = bottom[2]->cpu_data();

	Dtype* top_rois = top[0]->mutable_gpu_data();

	int height = bottom[0]->height();
	int width = bottom[0]->width();

	dim3 block(32, 16);
	dim3 grid(DIVUP(width, block.x), DIVUP(height, block.y));

	vector<int> shape;
	shape.push_back(num_anchors_);
	shape.push_back(4);
	shape.push_back(height);
	shape.push_back(width);
	shift_anchros_.Reshape(shape);
	int4* shifted_anchors_data = (int4*)shift_anchros_.mutable_gpu_data();
	const float4* anchors_data = (const float4*)anchors_.gpu_data();
	//double et = getTimeOfMSeconds();
	//LOG(INFO) << "P1 use time " << et-st << " ms";
	//st = getTimeOfMSeconds();
	EnumerateShiftedAnchors<<<grid, block>>>(height,
			width,
			num_anchors_,
			anchors_data,
			shifted_anchors_data,
			feat_stride_);
	cudaDeviceSynchronize();
	//et = getTimeOfMSeconds();
	//LOG(INFO) << "P2 use time " << et-st << " ms";

	//st = getTimeOfMSeconds();
	shape.clear();
	shape.push_back(1);
	shape.push_back(num_anchors_);
	shape.push_back(height);
	shape.push_back(width);
	scores_kept_.Reshape(shape);
	float* scores_kept_data = scores_kept_.mutable_gpu_data();

	int im_height = bottom_im_info[0];
	int im_width = bottom_im_info[1];
	float im_min_size = min_size * bottom_im_info[2];
	shape.clear();
	shape.push_back(1);
	shape.push_back(4*num_anchors_);
	shape.push_back(height);
	shape.push_back(width);
	proposals_.Reshape(shape);
	float4* proposals_data = (float4*)proposals_.mutable_gpu_data();
	//et = getTimeOfMSeconds();
	//LOG(INFO) << "P3 use time " << et-st << " ms";
	//st = getTimeOfMSeconds();
	EnumerateProposal<<<grid, block>>>(
			height,
			width,
			num_anchors_, 
			im_min_size, 
			im_height, 
			im_width,
			shifted_anchors_data,
			(const float*)bottom_bbox_deltas,
			(const float*)bottom_scores,
			proposals_data,
			scores_kept_data);
	cudaDeviceSynchronize();
	//et = getTimeOfMSeconds();
	//LOG(INFO) << "P4 use time " << et-st << " ms";

	//st = getTimeOfMSeconds();
	const int nSize = height*width*num_anchors_;
	thrust::device_vector<float> ret_scores_kept(
			scores_kept_data, scores_kept_data+nSize);
	thrust::device_vector<float4> ret_bbox(
			proposals_data, proposals_data+nSize);
	thrust::sort_by_key(ret_scores_kept.begin(), 
			ret_scores_kept.end(), ret_bbox.begin(), thrust::greater<float>());
	cudaDeviceSynchronize();
	//et = getTimeOfMSeconds();
	//LOG(INFO) << "P5 use time " << et-st << " ms";

	//st = getTimeOfMSeconds();
	proposals_data = thrust::raw_pointer_cast(&ret_bbox[0]);
	int* keep_out = new int[pre_nms_topN];
	int num_out = 0;
	if (pre_nms_topN > 0) {
		NMS_gpu(keep_out, &num_out, proposals_data, pre_nms_topN, nms_thresh);
	} else {
		NMS_gpu(keep_out, &num_out, proposals_data, nSize, nms_thresh);
	}
	if (post_nms_topN > 0 && num_out > post_nms_topN) {
		num_out = post_nms_topN;
	}
	cudaDeviceSynchronize();
	//et = getTimeOfMSeconds();
	//LOG(INFO) << "P6 use time " << et-st << " ms";

	//st = getTimeOfMSeconds();
	int* dev_keep_out = NULL;
	CUDA_CHECK( cudaMalloc((void**)&dev_keep_out, sizeof(int)*num_out) );
	CUDA_CHECK( cudaMemcpy(dev_keep_out, keep_out, 
		sizeof(int)*num_out, cudaMemcpyHostToDevice) );
	ProposalForward<<<DIVUP(num_out, 512), 512>>>(
			num_out, dev_keep_out, proposals_data, top_rois);

	//et = getTimeOfMSeconds();
	//LOG(INFO) << "P7 use time " << et-st << " ms";
	delete[] keep_out;
	CUDA_CHECK( cudaFree(dev_keep_out) );
	CUDA_POST_KERNEL_CHECK;
}



template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
	return Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(FrcnnProposalLayer);

} // namespace frcnn

} // namespace caffe
