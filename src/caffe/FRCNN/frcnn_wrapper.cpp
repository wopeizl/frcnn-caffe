#include "caffe/FRCNN/frcnn_wrapper.hpp"

namespace FRCNN_API {

    void Frcnn_wrapper::Set_Model(std::string &proto_file, std::string &model_file, std::string default_config) {
        FrcnnParam::load_param(default_config);
        net_.reset(new Net<float>(proto_file, caffe::TEST));
        net_->CopyTrainedLayersFrom(model_file);
        DLOG(INFO) << "SET MODEL DONE";
        caffe::Frcnn::FrcnnParam::print_param();
    }

    bool Frcnn_wrapper::prepare(const cv::Mat &input, cv::Mat &img) {
        //CHECK(FrcnnParam::test_scales.size() == 1) << "Only single-image batch implemented";

        float scale_factor = caffe::Frcnn::get_scale_factor(input.cols, input.rows, FrcnnParam::test_scales[0], FrcnnParam::test_max_size);

        const int height = input.rows;
        const int width = input.cols;
        DLOG(INFO) << "height: " << height << " width: " << width;
        input.convertTo(img, CV_32FC3);
        for (int r = 0; r < img.rows; r++) {
            for (int c = 0; c < img.cols; c++) {
                int offset = (r * img.cols + c) * 3;
                reinterpret_cast<float *>(img.data)[offset + 0] -= FrcnnParam::pixel_means[0]; // B
                reinterpret_cast<float *>(img.data)[offset + 1] -= FrcnnParam::pixel_means[1]; // G
                reinterpret_cast<float *>(img.data)[offset + 2] -= FrcnnParam::pixel_means[2]; // R
            }
        }
        cv::resize(img, img, cv::Size(), scale_factor, scale_factor);

        return true;
    }

    bool Frcnn_wrapper::batch_preprocess(const vector < cv::Mat > &img_in, vector<boost::shared_ptr<Blob<float> > >& input) {
        int nmat = img_in.size();
        boost::shared_ptr<Blob<float> > blob_pointer(new Blob<float>());
        blob_pointer->Reshape(nmat, img_in.at(0).channels(), img_in.at(0).rows, img_in.at(0).cols);

        float *blob_data = blob_pointer->mutable_cpu_data();
        const int cols = img_in.at(0).cols;
        const int rows = img_in.at(0).rows;
        int offset = img_in.at(0).channels() * img_in.at(0).rows * img_in.at(0).cols;
        for (int i = 0; i < nmat; i++) {
            cv::Mat img = img_in.at(i);
            if (i > 0) {
                CHECK_EQ(img.channels(), img_in.at(0).channels()) << "all image need same channel";
                CHECK_EQ(img.rows, img_in.at(0).rows) << "all image need same height!";
                CHECK_EQ(img.cols, img_in.at(0).cols) << "all image need same width!";
            }
            float* ptr = blob_data + i * offset;
            CHECK(img.isContinuous()) << "Warning : cv::Mat img_out is not Continuous !";
            DLOG(ERROR) << "img (CHW) : " << img.channels() << ", " << img.rows << ", " << img.cols;
            for (int j = 0; j < cols * rows; j++) {
                ptr[cols * rows * 0 + j] =
                    reinterpret_cast<float*>(img.data)[j * 3 + 0];// mean_[0];
                ptr[cols * rows * 1 + j] =
                    reinterpret_cast<float*>(img.data)[j * 3 + 1];// mean_[1];
                ptr[cols * rows * 2 + j] =
                    reinterpret_cast<float*>(img.data)[j * 3 + 2];// mean_[2];
            }
        }
        input.push_back(blob_pointer);

        return true;
    }

    bool Frcnn_wrapper::copySingleBlob(vector<boost::shared_ptr<Blob<float> > >& input, int i, vector<boost::shared_ptr<Blob<float> > >& output) {
        for (vector<boost::shared_ptr<Blob<float> > >::iterator iter = input.begin(); iter != input.end(); ++iter) {
            boost::shared_ptr<Blob<float> > cb = *iter;
            if (i >= cb->num()) {
                return false;
            }
            boost::shared_ptr<Blob<float> > blob_pointer(new Blob<float>());
            blob_pointer->Reshape(1, cb->channels(), cb->height(), cb->width());

            //caffe::caffe_copy()

            float *cb_data = cb->mutable_cpu_data();
            float *blob_data = blob_pointer->mutable_cpu_data();

            std::memcpy(blob_data, (char*)cb_data + sizeof(float) * i * cb->channels() * cb->height() * cb->width(),
                sizeof(float) * cb->channels() * cb->height() * cb->width());

            output.push_back(blob_pointer);
        }

        return true;
    }

    bool Frcnn_wrapper::preprocess(const cv::Mat &img, vector<boost::shared_ptr<Blob<float> > >& input) {
        float scale_factor = caffe::Frcnn::get_scale_factor(img.cols, img.rows, FrcnnParam::test_scales[0], FrcnnParam::test_max_size);

        boost::shared_ptr<Blob<float> > blob_pointer(new Blob<float>());
        CHECK(img.isContinuous()) << "Warning : cv::Mat img_out is not Continuous !";
        DLOG(ERROR) << "img (CHW) : " << img.channels() << ", " << img.rows << ", " << img.cols;
        blob_pointer->Reshape(1, img.channels(), img.rows, img.cols);
        float *blob_data = blob_pointer->mutable_cpu_data();
        const int cols = img.cols;
        const int rows = img.rows;
        for (int i = 0; i < cols * rows; i++) {
            blob_data[cols * rows * 0 + i] =
                reinterpret_cast<float*>(img.data)[i * 3 + 0];// mean_[0]; 
            blob_data[cols * rows * 1 + i] =
                reinterpret_cast<float*>(img.data)[i * 3 + 1];// mean_[1];
            blob_data[cols * rows * 2 + i] =
                reinterpret_cast<float*>(img.data)[i * 3 + 2];// mean_[2];
        }
        input.push_back(blob_pointer);

        std::vector<float> im_info(3);
        im_info[0] = img.rows;
        im_info[1] = img.cols;
        im_info[2] = scale_factor;
        DLOG(ERROR) << "im_info : " << im_info[0] << ", " << im_info[1] << ", " << im_info[2];
        boost::shared_ptr<Blob<float> > blob_pointer1(new Blob<float>());
        //const vector<Blob<float> *> &input_blobs = net_->input_blobs();
        blob_pointer1->Reshape(1, im_info.size(), 1, 1);
        blob_data = blob_pointer1->mutable_cpu_data();
        std::memcpy(blob_data, &im_info[0], sizeof(float) * im_info.size());
        input.push_back(blob_pointer1);

        return true;
    }

    bool Frcnn_wrapper::preprocess(const cv::Mat &img_in, const int blob_idx) {
        const vector<Blob<float> *> &input_blobs = net_->input_blobs();
        CHECK(img_in.isContinuous()) << "Warning : cv::Mat img_out is not Continuous !";
        DLOG(ERROR) << "img_in (CHW) : " << img_in.channels() << ", " << img_in.rows << ", " << img_in.cols;
        input_blobs[blob_idx]->Reshape(1, img_in.channels(), img_in.rows, img_in.cols);
        float *blob_data = input_blobs[blob_idx]->mutable_cpu_data();
        const int cols = img_in.cols;
        const int rows = img_in.rows;
        for (int i = 0; i < cols * rows; i++) {
            blob_data[cols * rows * 0 + i] =
                reinterpret_cast<float*>(img_in.data)[i * 3 + 0];// mean_[0]; 
            blob_data[cols * rows * 1 + i] =
                reinterpret_cast<float*>(img_in.data)[i * 3 + 1];// mean_[1];
            blob_data[cols * rows * 2 + i] =
                reinterpret_cast<float*>(img_in.data)[i * 3 + 2];// mean_[2];
        }
        return true;
    }

    bool Frcnn_wrapper::preprocess(std::vector<cv::Mat >& img_in, const int blob_idx) {
        int nmat = img_in.size();

        const vector<Blob<float> *> &input_blobs = net_->input_blobs();
        input_blobs[blob_idx]->Reshape(nmat, img_in.at(0).channels(), img_in.at(0).rows, img_in.at(0).cols);

        float *blob_data = input_blobs[blob_idx]->mutable_cpu_data();
        const int cols = img_in.at(0).cols;
        const int rows = img_in.at(0).rows;
        int offset = img_in.at(0).channels() * img_in.at(0).rows * img_in.at(0).cols;

        for (int i = 0; i < nmat; i++) {
            cv::Mat img = img_in.at(i);

            if (i > 0) {
                CHECK_EQ(img.channels(), img_in.at(0).channels()) << "all image need same channel";
                CHECK_EQ(img.rows, img_in.at(0).rows) << "all image need same height!";
                CHECK_EQ(img.cols, img_in.at(0).cols) << "all image need same width!";
            }

            float* ptr = blob_data + i * offset;

            CHECK(img.isContinuous()) << "Warning : cv::Mat img_out is not Continuous !";
            DLOG(ERROR) << "img (CHW) : " << img.channels() << ", " << img.rows << ", " << img.cols;

            for (int j = 0; j < cols * rows; j++) {
                ptr[cols * rows * 0 + j] =
                    reinterpret_cast<float*>(img.data)[j * 3 + 0];// mean_[0];
                ptr[cols * rows * 1 + j] =
                    reinterpret_cast<float*>(img.data)[j * 3 + 1];// mean_[1];
                ptr[cols * rows * 2 + j] =
                    reinterpret_cast<float*>(img.data)[j * 3 + 2];// mean_[2];
            }
        }
        return true;
    }

    bool Frcnn_wrapper::preprocess(const int nmat, const vector<float> &data, const int blob_idx) {
        const vector<Blob<float> *> &input_blobs = net_->input_blobs();
        input_blobs[blob_idx]->Reshape(nmat, data.size(), 1, 1);
        float *blob_data = input_blobs[blob_idx]->mutable_cpu_data();

        for (int i = 0; i < nmat; i++) {
            std::memcpy(blob_data, &data[0], sizeof(float) * data.size());
            blob_data += data.size();
        }
        return true;
    }

    bool Frcnn_wrapper::predict_b(const std::vector<cv::Mat >& img_in, vector<boost::shared_ptr<Blob<float> > >& output) {
        if (img_in.size() < 1) {
            LOG(WARNING) << "Empty image input vector?";
            return false;
        }

        int nmat_ = img_in.size();
        /// if(nmat_ == 1)return predict(img_in.at(0), results);

        float scale_factor = caffe::Frcnn::get_scale_factor(img_in.at(0).cols, img_in.at(0).rows, FrcnnParam::test_scales[0], FrcnnParam::test_max_size);
        std::vector<cv::Mat> dst_img;
        const int height = img_in.at(0).rows;
        const int width = img_in.at(0).cols;
        DLOG(INFO) << "height: " << height << " width: " << width;

        for (int i = 0; i < nmat_; i++) {
            cv::Mat img;
            img_in.at(i).convertTo(img, CV_32FC3);

            for (int r = 0; r < img.rows; r++) {
                for (int c = 0; c < img.cols; c++) {
                    int offset = (r * img.cols + c) * 3;
                    reinterpret_cast<float *>(img.data)[offset + 0] -= FrcnnParam::pixel_means[0]; // B
                    reinterpret_cast<float *>(img.data)[offset + 1] -= FrcnnParam::pixel_means[1]; // G
                    reinterpret_cast<float *>(img.data)[offset + 2] -= FrcnnParam::pixel_means[2]; // R
                }
            }
            cv::resize(img, img, cv::Size(), scale_factor, scale_factor);
            dst_img.push_back(img);
        }

        std::vector<float> im_info(4);
        im_info[0] = dst_img.at(0).rows;
        im_info[1] = dst_img.at(0).cols;
        im_info[2] = scale_factor;
        im_info[3] = scale_factor;

        DLOG(ERROR) << "im_info : " << im_info[0] << ", " << im_info[1] << ", " << im_info[2];
        this->preprocess(dst_img, 0);
        this->preprocess(nmat_, im_info, 1);

        vector<std::string> blob_names(3);
        blob_names[0] = "rois";
        blob_names[1] = "cls_prob";
        blob_names[2] = "bbox_pred";

        float loss;
        net_->Forward(&loss);
        for (int i = 0; i < blob_names.size(); ++i) {
            output.push_back(this->net_->blob_by_name(blob_names[i]));
        }

        return true;
    }

    bool Frcnn_wrapper::postprocess_b(const std::vector<cv::Mat >& img_in, int imat, vector<boost::shared_ptr<Blob<float> > >& input, std::vector<caffe::Frcnn::BBox<float> > &results) {

        if (imat < 0 || imat >= img_in.size()) {
            return false;
        }

        int nmat_ = img_in.size();
        float scale_factor = caffe::Frcnn::get_scale_factor(img_in[imat].cols, img_in[imat].rows, FrcnnParam::test_scales[0], FrcnnParam::test_max_size);
        boost::shared_ptr<Blob<float> > rois(input[0]);
        boost::shared_ptr<Blob<float> > cls_prob(input[1]);
        boost::shared_ptr<Blob<float> > bbox_pred(input[2]);
        const int height = img_in.at(0).rows;
        const int width = img_in.at(0).cols;

        const int box_num = bbox_pred->num();
        const int cls_num = cls_prob->channels();

        CHECK_EQ(box_num, cls_prob->num()) << "BBox num != cls num?";
        CHECK_EQ(box_num, rois->num()) << "Rois num != BBox num?";
        CHECK_EQ(box_num % nmat_, 0) << " BBox num % nmat != 0?";

        int box_num_per_img = box_num / nmat_;
        results.clear();

        const float* rois_blob_ptr = NULL;
        const float* bbox_data_ptr = NULL;
        const float* cls_prob_ptr = NULL;

        rois_blob_ptr = rois->cpu_data() + rois->offset(imat * box_num_per_img);
        bbox_data_ptr = bbox_pred->cpu_data() + bbox_pred->offset(imat * box_num_per_img);
        cls_prob_ptr = cls_prob->cpu_data() + cls_prob->offset(imat * box_num_per_img);

        for (int cls = 1; cls < cls_num; cls++) {
            vector<BBox<float> > bbox;
            for (int i = 0; i < box_num_per_img; i++) {
                float score = cls_prob_ptr[i * cls_num + cls];
                /// Bug??
                /// CHECK_EQ((int)(rois_blob_ptr[(i * 5) + 0]), imat) << "ROIS No.0 value should equal mat id!";
                Point4f<float> roi(rois_blob_ptr[(i * 5) + 1] / scale_factor,
                    rois_blob_ptr[(i * 5) + 2] / scale_factor,
                    rois_blob_ptr[(i * 5) + 3] / scale_factor,
                    rois_blob_ptr[(i * 5) + 4] / scale_factor);

                Point4f<float> delta(bbox_data_ptr[(i * cls_num + cls) * 4 + 0],
                    bbox_data_ptr[(i * cls_num + cls) * 4 + 1],
                    bbox_data_ptr[(i * cls_num + cls) * 4 + 2],
                    bbox_data_ptr[(i * cls_num + cls) * 4 + 3]);

                Point4f<float> box = caffe::Frcnn::bbox_transform_inv(roi, delta);
                box[0] = std::max(0.0f, box[0]);
                box[1] = std::max(0.0f, box[1]);
                box[2] = std::min(width - 1.f, box[2]);
                box[3] = std::min(height - 1.f, box[3]);

                bbox.push_back(BBox<float>(box, score, cls));
            }

            sort(bbox.begin(), bbox.end());

            vector<bool> select(box_num_per_img, true);
            // Apply NMS
            for (int i = 0; i < box_num_per_img; i++) {
                if (select[i]) {
                    if (bbox[i].confidence < FrcnnParam::test_score_thresh) break;
                    for (int j = i + 1; j < box_num_per_img; j++) {
                        if (select[j]) {
                            if (get_iou(bbox[i], bbox[j]) > FrcnnParam::test_nms) {
                                select[j] = false;
                            }
                        }
                    }
                    results.push_back(bbox[i]);
                }
            }
        }

        return true;
    }

    bool Frcnn_wrapper::predict(vector<boost::shared_ptr<Blob<float> > >& input, vector<boost::shared_ptr<Blob<float> > >& output) {
        vector<std::string> blob_names(3);
        blob_names[0] = "rois";
        blob_names[1] = "cls_prob";
        blob_names[2] = "bbox_pred";

        DLOG(ERROR) << "FORWARD BEGIN";
        float loss;

        const vector<Blob<float>*>& input_blobs = net_->input_blobs();
        for (int i = 0; i < input.size(); ++i) {
            input_blobs[i]->Reshape(*input[i].get());
        }

        net_->Forward(&loss);

        for (int i = 0; i < blob_names.size(); ++i) {
            output.push_back(this->net_->blob_by_name(blob_names[i]));
        }
        DLOG(ERROR) << "FORWARD END, Loss : " << loss;

        return true;
    }

    bool Frcnn_wrapper::postprocess(const cv::Mat &img, vector<boost::shared_ptr<Blob<float> > >& input, std::vector<caffe::Frcnn::BBox<float> > &output) {
        float scale_factor = caffe::Frcnn::get_scale_factor(img.cols, img.rows, FrcnnParam::test_scales[0], FrcnnParam::test_max_size);
        const int height = img.rows;
        const int width = img.cols;

        boost::shared_ptr<Blob<float> > rois(input[0]);
        boost::shared_ptr<Blob<float> > cls_prob(input[1]);
        boost::shared_ptr<Blob<float> > bbox_pred(input[2]);

        const int box_num = bbox_pred->num();
        const int cls_num = cls_prob->channels();
        CHECK_EQ(cls_num, caffe::Frcnn::FrcnnParam::n_classes);
        output.clear();

        for (int cls = 1; cls < cls_num; cls++) {
            vector<BBox<float> > bbox;
            for (int i = 0; i < box_num; i++) {
                float score = cls_prob->cpu_data()[i * cls_num + cls];

                Point4f<float> roi(rois->cpu_data()[(i * 5) + 1] / scale_factor,
                    rois->cpu_data()[(i * 5) + 2] / scale_factor,
                    rois->cpu_data()[(i * 5) + 3] / scale_factor,
                    rois->cpu_data()[(i * 5) + 4] / scale_factor);

                Point4f<float> delta(bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 0],
                    bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 1],
                    bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 2],
                    bbox_pred->cpu_data()[(i * cls_num + cls) * 4 + 3]);

                Point4f<float> box = caffe::Frcnn::bbox_transform_inv(roi, delta);
                box[0] = std::max(0.0f, box[0]);
                box[1] = std::max(0.0f, box[1]);
                box[2] = std::min(width - 1.f, box[2]);
                box[3] = std::min(height - 1.f, box[3]);

                // BBox tmp(box, score, cls);
                // LOG(ERROR) << "cls: " << tmp.id << " score: " << tmp.confidence;
                // LOG(ERROR) << "roi: " << roi.to_string();
                bbox.push_back(BBox<float>(box, score, cls));
            }
            sort(bbox.begin(), bbox.end());
            vector<bool> select(box_num, true);
            // Apply NMS
            for (int i = 0; i < box_num; i++)
                if (select[i]) {
                    if (bbox[i].confidence < FrcnnParam::test_score_thresh) break;
                    for (int j = i + 1; j < box_num; j++) {
                        if (select[j]) {
                            if (get_iou(bbox[i], bbox[j]) > FrcnnParam::test_nms) {
                                select[j] = false;
                            }
                        }
                    }
                    output.push_back(bbox[i]);
                }
        }

        return true;
    }


} // FRCNN_API
