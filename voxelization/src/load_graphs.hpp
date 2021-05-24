/** TODO:: Add doxygen and macro to avoid including multiple header files */
#include <tuple>
/** Internal header files */
#include "pre_processing.hpp"


using tensor_vec = std::vector<tensorflow::Tensor>;
using tensor_tup = std::tuple<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::VectorXf>;

class LoadGraphAndPredict{
    public:
    /** Constructor */
    LoadGraphAndPredict(const std::string & bin_path);
    
    static constexpr auto RPN_SCORE_THRESH{0.96};
    
    void ApplyNMS();
    private:
    
    std::string bin_file;  
    public:
    tensor_vec LoadModel();
    tensorflow::Tensor LoadAnchor();
    tensor_tup LoadBoxes3D();
    Eigen::MatrixXf DeltaTo3DBoxes(const Eigen::MatrixXf & deltas, const Eigen::MatrixXf &anchors);

};