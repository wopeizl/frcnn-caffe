#include <vector>
#include <string>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/FRCNN/util/frcnn_param.hpp"
#include "caffe/FRCNN/util/frcnn_helper.hpp"

namespace FRCNN_API {

    using std::vector;
    using caffe::Blob;
    using caffe::Net;
    using caffe::Frcnn::FrcnnParam;
    using caffe::Frcnn::Point4f;
    using caffe::Frcnn::BBox;

    class Frcnn_wrapper {
    public:
        Frcnn_wrapper(std::string &proto_file, std::string &model_file, std::string default_config) {
            Set_Model(proto_file, model_file, default_config);
        }

        static bool prepare(const cv::Mat &input, cv::Mat &img);
        static bool preprocess(const cv::Mat &img, vector<boost::shared_ptr<Blob<float> > >& input);
        static bool copySingleBlob(vector<boost::shared_ptr<Blob<float> > >& input, int i, vector<boost::shared_ptr<Blob<float> > >& output);
        static bool batch_preprocess(const vector < cv::Mat > &imgs, vector<boost::shared_ptr<Blob<float> > >& input);
        static bool postprocess(const cv::Mat &img, vector<boost::shared_ptr<Blob<float> > >& input, std::vector<caffe::Frcnn::BBox<float> > &output);
        bool predict(vector<boost::shared_ptr<Blob<float> > >& input, vector<boost::shared_ptr<Blob<float> > >& output);

    private:
        void Set_Model(std::string &proto_file, std::string &model_file, std::string default_config);
        boost::shared_ptr<Net<float> > net_;
    };

}
