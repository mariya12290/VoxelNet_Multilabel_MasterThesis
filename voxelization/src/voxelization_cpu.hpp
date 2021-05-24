/** Standard header files */
#include <unordered_map>
#include <utility>


/** Eigen header files */
#include <Eigen/Dense>

/** Point cloud header files */
#include <pcl/io/pcd_io.h>
#include <pcl/common/point_operators.h>
#include <pcl/common/io.h>
#include <pcl/search/organized.h>

/** Tensorflow header files */
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

using tensor_dict = std::vector<std::pair<std::string,tensorflow::Tensor>>;
using Key = std::tuple<int,int,int>;

/** Hash key to make tuple a key in map */

struct KeyHash {
    std::size_t operator()(const Key & key) const
    {
        return boost::hash_value(key);
    }
};

class PreProcessing{
    public:
    /** Constructor */
    PreProcessing(const std::string& bin_path);

    /** One can change the below values based on application */
    const Eigen::Vector3f lidar_coord = Eigen::Vector3f(0,20,3);
    const Eigen::Vector3f voxel_size = Eigen::Vector3f(0.4,0.2,0.2);
    const std::array<int,3>grid_size{10,200,240};
    
    const std::string bin_file{""};
    tensor_dict PointCloudProcess();
   
   private:
   pcl::PointCloud<pcl::PointXYZI> points;
   int number_of_points{};
   
    void ReadPointCloud();
    Eigen::MatrixXi EigenRowSort(Eigen::MatrixXi& mat);
    
};

