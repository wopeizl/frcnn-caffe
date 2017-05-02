// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include "caffe/FRCNN/frcnn_proposal_layer.hpp"
#include "caffe/FRCNN/util/frcnn_utils.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"  

namespace caffe {

namespace Frcnn {

using std::vector;
using namespace std;

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {

    feat_stride_ = FrcnnParam::feat_stride;
    CHECK_GT(feat_stride_, 0);
    LOG(INFO) << "Feature stride: " << feat_stride_;
    // default values to compute anchors
    int base_size = 16;
    float arr0[] = { 0.5, 1, 2 };
    vector<float> ratios(arr0, arr0 + sizeof(arr0) / sizeof(arr0[0]));
    int arr1[] = { 8, 16, 32 };
    vector<int> scales(arr1, arr1 + sizeof(arr1) / sizeof(arr1[0]));

    vector<vector<float> > anchors = generate_anchors(base_size, ratios, scales);
    LOG(INFO) << "Anchors: ";
    for (int i = 0; i < anchors.size(); ++i) {
        LOG(INFO) << anchors[i][0] << " " << anchors[i][1]
            << " " << anchors[i][2] << " " << anchors[i][3];
    }
    num_anchors_ = anchors.size();
    anchors_.Reshape(1, 1, 1, 4 * num_anchors_);
    float* anchors_data = anchors_.mutable_cpu_data();
    for (int i = 0; i < num_anchors_; ++i) {
        for (int j = 0; j < 4; ++j) {
            anchors_data[i * 4 + j] = anchors[i][j];
        }
    }
}

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    int post_nms_topN = 0;
    if (this->phase_ == TRAIN) {
        post_nms_topN = FrcnnParam::rpn_post_nms_top_n;
    }
    else {
        post_nms_topN = FrcnnParam::test_rpn_post_nms_top_n;
    }

    vector<int> roi_shape;
    roi_shape.push_back(post_nms_topN);
    roi_shape.push_back(5);
    top[0]->Reshape(roi_shape);
}

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
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

    const Dtype* bottom_scores = bottom[0]->cpu_data();
    const Dtype* bottom_bbox_deltas = bottom[1]->cpu_data();
    const Dtype* bottom_im_info = bottom[2]->cpu_data();

    Dtype* top_rois = top[0]->mutable_cpu_data();
    caffe_set(top[0]->count(), Dtype(0), top_rois);

    // 1. Generate proposals from bbox deltas and shifted anchors
    int height = bottom[0]->height();
    int width = bottom[0]->width();

    LOG(INFO) << "score map size: " << height << "x" << width;
    //CPUTimer timer;
    //timer.Start();
    //double part_time = 0;

    // enumerate all shifts 
    vector<int> shift_x, shift_y;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            shift_x.push_back(j * feat_stride_);
            shift_y.push_back(i * feat_stride_);
        }
    }

    vector<vector<int> > shifts;
    vector<int> single_shift;
    int K = shift_x.size();

    for (int i = 0; i < K; ++i)
    {
        single_shift.clear();
        single_shift.push_back(shift_x[i]);
        single_shift.push_back(shift_y[i]);
        single_shift.push_back(shift_x[i]);
        single_shift.push_back(shift_y[i]);
        shifts.push_back(single_shift);
    }
    int A = num_anchors_;
    vector<vector<int> > shifted_anchors; // sas
    const float* anchors_data = anchors_.cpu_data();
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < A; ++j) {
            vector<int> single_sa;
            for (int k = 0; k < 4; ++k) {
                single_sa.push_back(anchors_data[j * 4 + k] + shifts[i][k]);
            }
            shifted_anchors.push_back(single_sa);
            // LOG(INFO) << single_sa[0] << "," << single_sa[1] << "," << single_sa[2] << "," << single_sa[3];
        }
    }

    vector<vector<Dtype> > deltas;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int a = 0; a < num_anchors_; ++a) {
                vector<Dtype> single_ds;
                for (int k = 0; k < 4; ++k) {
                    // const int index = a*4 + w * num_anchors_*4 + h*width*num_anchors_*4;
                    const int index = w + h*width + (a * 4 + k)*(height*width);
                    single_ds.push_back(bottom_bbox_deltas[index]);
                }
                deltas.push_back(single_ds);
            }
        }
    }

    vector<Dtype> scores, scores_kept;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int a = 0; a < num_anchors_; ++a) {
                const int index = w + h*width + (num_anchors_ + a)*(height*width);
                scores.push_back(bottom_scores[index]);
            }
        }
    }
    //std::cout << std::endl;

    //part_time += timer.MicroSeconds();
    //std::cout << "P1: elapsed time: " << part_time << std::endl;
    //part_time = 0; timer.Start();

    int im_height, im_width;
    im_height = bottom_im_info[0];
    im_width = bottom_im_info[1];
    // LOG(INFO) << "im height: " << im_height << " im width: " << im_width;
    LOG(INFO) << "im_size: (" << im_height << ", " << im_width << ")";
    LOG(INFO) << "scale: " << bottom_im_info[2];
    float im_min_size = min_size * bottom_im_info[2];

    CHECK_EQ(shifted_anchors.size(), deltas.size());
    vector<vector<float> > proposals;
    for (int i = 0; i < shifted_anchors.size(); ++i) {
        float ctr_x, ctr_y, w, h;
        float pred_ctr_x, pred_ctr_y, pred_w, pred_h;
        vector<float> box;
        w = shifted_anchors[i][2] - shifted_anchors[i][0] + 1.0;
        h = shifted_anchors[i][3] - shifted_anchors[i][1] + 1.0;
        ctr_x = shifted_anchors[i][0] + 0.5*w;
        ctr_y = shifted_anchors[i][1] + 0.5*h;

        pred_ctr_x = deltas[i][0] * w + ctr_x;
        pred_ctr_y = deltas[i][1] * h + ctr_y;
        pred_w = exp(deltas[i][2]) * w;
        pred_h = exp(deltas[i][3]) * h;

        // 2. clip predicted boxes to image
        float x1, y1, x2, y2;
        x1 = max(min(pred_ctr_x - 0.5 * pred_w, im_width - 1.0), 0.0);
        y1 = max(min(pred_ctr_y - 0.5 * pred_h, im_height - 1.0), 0.0);
        x2 = max(min(pred_ctr_x + 0.5 * pred_w, im_width - 1.0), 0.0);
        y2 = max(min(pred_ctr_y + 0.5 * pred_h, im_height - 1.0), 0.0);

        // 3. remove predicted boxes with either height or width < threshold
        if ((x2 - x1 + 1 >= im_min_size) && (y2 - y1 + 1 >= im_min_size))
        {
            box.clear();
            box.push_back(x1);
            box.push_back(y1);
            box.push_back(x2);
            box.push_back(y2);
            proposals.push_back(box);
            scores_kept.push_back(scores[i]);
            // LOG(INFO) << box[0] << "," << box[1] << "," << box[2] << "," << box[3];
        }
    }

    // 4. sort all: take the pre_nms_topN
    vector<data> proposal_pairs(proposals.size());
    for (int i = 0; i < proposals.size(); ++i) {
        proposal_pairs[i].bbox = proposals[i];
        proposal_pairs[i].score = (float)scores_kept[i];
        proposal_pairs[i].index = i;
    }

    std::sort(proposal_pairs.begin(), proposal_pairs.end(), by_score());

    // 5. take top pre_nms_topN (e.g. 6000)
    if (pre_nms_topN > 0) {
        // proposal_pairs.erase(proposal_pairs.begin() + pre_nms_topN, proposal_pairs.end());
        proposal_pairs.resize(pre_nms_topN);
    }

    //part_time += timer.MicroSeconds();
    //std::cout << "P2: elapsed time: " << part_time << std::endl;
    //part_time = 0; timer.Start();

    // 6. apply nms (e.g. threshold = 0.7)
    //nms_thresh
    vector<int> keep = nms_cpu(proposal_pairs, nms_thresh);

    //part_time += timer.MicroSeconds();
    //std::cout << "P3: elapsed time: " << part_time << std::endl;
    //part_time = 0; timer.Start();

    // 7. take post_nms_topN (e.g. 300)
    // 8. return the top proposals (-> RoIs top)
    if (post_nms_topN > 0 && keep.size() > post_nms_topN) {
        keep.resize(post_nms_topN);
    }

    for (int i = 0; i < keep.size(); ++i) {
        top_rois[5 * i] = 0.0;
        for (int k = 0; k < 4; ++k) {
            top_rois[5 * i + 1 + k] = proposal_pairs[keep[i]].bbox[k];
        }
        // LOG(INFO) << top_rois[5*i + 1] << " " << top_rois[5*i + 2] << " " << top_rois[5*i + 3] << " " << top_rois[5*i + 4];
    }

    // if (top.size() >= 2) {
    // 	Dtype* top_scores = top[1]->mutable_cpu_data();
    // 	caffe_set(top[1]->count(), Dtype(0), top_scores);

    // 	for (int i = 0; i < keep.size(); ++i) {
    // 		top_scores[i] = proposal_pairs[keep[i]].score;
    // 	}
    // 	top[1]->Reshape(1, A, height, width);
    // }
    }

template <typename Dtype>
void FrcnnProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
      NOT_IMPLEMENTED;
}

template <typename Dtype>
vector<int> FrcnnProposalLayer<Dtype>::nms_cpu(vector<data> dets, float thresh)
{
    int ndets = dets.size();
    vector<int> suppressed(ndets);
    for (int i = 0; i < ndets; ++i) {
        dets[i].index = i;
        suppressed[i] = 0;
    }
    // sort(dets.begin(), dets.end(), by_score());
    vector<int> order(ndets);
    for (int i = 0; i < ndets; ++i) {
        order[i] = dets[i].index;
    }
    vector<int> keep;
    int i, j;
    float ix1, iy1, ix2, iy2, iarea;
    float jx1, jy1, jx2, jy2, jarea;
    float xx1, yy1, xx2, yy2, w, h, inter, ovr;
    for (int _i = 0; _i < ndets; ++_i) {
        i = order[_i];
        if (suppressed[i] == 1) continue;
        keep.push_back(i);
        ix1 = dets[i].bbox[0];
        iy1 = dets[i].bbox[1];
        ix2 = dets[i].bbox[2];
        iy2 = dets[i].bbox[3];
        iarea = (ix2 - ix1 + 1) * (iy2 - iy1 + 1);
        for (int _j = _i + 1; _j < ndets; ++_j) {
            j = order[_j];
            if (suppressed[j] == 1) continue;
            jx1 = dets[j].bbox[0];
            jy1 = dets[j].bbox[1];
            jx2 = dets[j].bbox[2];
            jy2 = dets[j].bbox[3];
            xx1 = max(ix1, jx1);
            yy1 = max(iy1, jy1);
            xx2 = min(ix2, jx2);
            yy2 = min(iy2, jy2);
            w = max(float(0.), xx2 - xx1 + 1);
            h = max(float(0.), yy2 - yy1 + 1);
            jarea = (jx2 - jx1 + 1) * (jy2 - jy1 + 1);
            inter = w * h;
            ovr = inter / (iarea + jarea - inter);
            if (ovr >= thresh) suppressed[j] = 1;
        }
    }
    return keep;
}

template <typename Dtype>
vector<vector<float> > FrcnnProposalLayer<Dtype>::_ratio_enum(vector<int> base_anchor, vector<float> ratios)
{
    int w = base_anchor[2] - base_anchor[0] + 1;
    int h = base_anchor[3] - base_anchor[1] + 1;
    float x_ctr = base_anchor[0] + 0.5*(w - 1);
    float y_ctr = base_anchor[1] + 0.5*(h - 1);
    int size = w * h;
    float wi, hi;
    vector<vector<float> > ratio_anchors(ratios.size());
    for (int i = 0; i < ratios.size(); ++i)
    {
        wi = round(sqrt(1.0*size / ratios[i]));
        hi = round(wi*ratios[i]);
        ratio_anchors[i].push_back(x_ctr - 0.5*(wi - 1));
        ratio_anchors[i].push_back(y_ctr - 0.5*(hi - 1));
        ratio_anchors[i].push_back(x_ctr + 0.5*(wi - 1));
        ratio_anchors[i].push_back(y_ctr + 0.5*(hi - 1));
    }
    // LOG(INFO) << "ratio size: " << ratios.size();
    return ratio_anchors;
}

template <typename Dtype>
vector<vector<float> > FrcnnProposalLayer<Dtype>::_scale_enum(vector<float> base_anchor, vector<int> scales)
{
    CHECK_EQ(base_anchor.size(), 4);
    float w = base_anchor[2] - base_anchor[0] + 1;
    float h = base_anchor[3] - base_anchor[1] + 1;
    float x_ctr = base_anchor[0] + 0.5*(w - 1);
    float y_ctr = base_anchor[1] + 0.5*(h - 1);
    float wi, hi;
    vector<vector<float> > scale_anchors(scales.size());
    for (int i = 0; i < scales.size(); ++i)
    {
        wi = w * scales[i];
        hi = h * scales[i];
        scale_anchors[i].push_back(x_ctr - 0.5*(wi - 1));
        scale_anchors[i].push_back(y_ctr - 0.5*(hi - 1));
        scale_anchors[i].push_back(x_ctr + 0.5*(wi - 1));
        scale_anchors[i].push_back(y_ctr + 0.5*(hi - 1));
    }
    // LOG(INFO) << "scale size: " << scales.size();
    return scale_anchors;
}

template <typename Dtype>
vector<vector<float> > FrcnnProposalLayer<Dtype>::generate_anchors(int base_size, vector<float> ratios, vector<int> scales)
{
    vector<int> base_anchor;
    vector<vector<float> > anchors;
    base_anchor.push_back(0);
    base_anchor.push_back(0);
    base_anchor.push_back(base_size - 1);
    base_anchor.push_back(base_size - 1);
    vector<vector<float> >  ratio_anchors = _ratio_enum(base_anchor, ratios);
    anchors.reserve(ratios.size() * scales.size());
    for (int i = 0; i < ratio_anchors.size(); ++i)
    {
        vector<vector<float> > scale_anchors = _scale_enum(ratio_anchors[i], scales);
        anchors.insert(anchors.end(), scale_anchors.begin(), scale_anchors.end());
        scale_anchors.clear();
    }
    ratio_anchors.clear();
    return anchors;
}


#ifdef CPU_ONLY
STUB_GPU(FrcnnProposalLayer);
#endif

INSTANTIATE_CLASS(FrcnnProposalLayer);
REGISTER_LAYER_CLASS(FrcnnProposal);

} // namespace frcnn

} // namespace caffe
