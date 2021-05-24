/** Std header files */
#include <memory>
#include <chrono>

/** Internal header files */
#include "load_graphs.hpp"
#include "load_model.hpp"
#include "utils.hpp"


LoadGraphAndPredict::LoadGraphAndPredict(const std::string& bin_path):bin_file{bin_path}{
    /**Do Something */
}

tensor_vec LoadGraphAndPredict::LoadModel(){
  /** TODO:: Make it to read just .pb file */
    std::unique_ptr<PreProcessing>ptr = std::make_unique<PreProcessing>(bin_file);
    /** finding the execution time */
    //auto start = std::chrono::high_resolution_clock::now();

    tensor_dict tensors = ptr->PointCloudProcess();

    // auto stop = std::chrono::high_resolution_clock::now();

    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    // std::cout<<duration.count()<<std::endl;

    

    tensorflow::Session *session;
    bool allow_growth {true};
    tensorflow::SessionOptions session_options;
    //session_options.config.mutable_gpu_options()->set_visible_device_list("3");
    session_options.config.set_allow_soft_placement(true);
    session_options.config.mutable_gpu_options()->set_allow_growth(true);
    tensorflow::Status status = tensorflow::NewSession(session_options, &session);
    
    /**  it is taking too much time, since it is running on CPU, make sure it will run on GPU 
     * and also .pb file of tensorflow version 1 is not loading on tensorflow version 2 */
    std::string graph_fn = "/home/surendra/tensorflow_cc/Object_Detection/saved_model/checkpoint-00006481.meta";
    std::string checkpoint_fn = "/home/surendra/tensorflow_cc/Object_Detection/saved_model/checkpoint-00006481";

    //auto start = std::chrono::high_resolution_clock::now();
    TF_CHECK_OK(LoadGraph(session, graph_fn, checkpoint_fn));
    //auto stop = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    //std::cout<<duration.count()<<std::endl;
    std::vector<tensorflow::Tensor> outputs;
    
    /** Change the GPU options depending upon use case or available gpu */
    TF_CHECK_OK(
        session->Run(tensors, {"gpu_1/MiddleAndRPN_/conv21/Conv2D:0","gpu_1/MiddleAndRPN_/prob:0"}, {}, &outputs));
    
    //std::cout << "output          " << outputs[0].DebugString() << std::endl; //[2,200,240,14]
  
    return outputs;

}

tensorflow::Tensor LoadGraphAndPredict::LoadAnchor(){

    /** TODO:: Just create one Session for all graph */
    /** TODO:: make it to load just .pb file */
    tensorflow::Session *session;

    bool allow_growth {true};
    tensorflow::SessionOptions session_options;
    auto start = std::chrono::high_resolution_clock::now();
    //session_options.config.mutable_gpu_options()->set_visible_device_list("3");
    session_options.config.set_allow_soft_placement(true);
    session_options.config.mutable_gpu_options()->set_allow_growth(true);
    tensorflow::Status status = tensorflow::NewSession(session_options, &session);

    /** Implementation of anchors generation on CPU or GPU might increase the inference time
     * or running the graph on GPU might also increase the inference time */
    std::string graph_fn = "/home/surendra/tensorflow_cc/Object_Detection/graphs/anchors/anchors.meta";
    std::string checkpoint_fn = "/home/surendra/tensorflow_cc/Object_Detection/graphs/anchors/anchors";

    TF_CHECK_OK(LoadGraph(session, graph_fn, checkpoint_fn));

    tensor_vec outputs;

    /** Prepare inputs */
    tensorflow::TensorShape data_shape({1});
    tensorflow::Tensor a(tensorflow::DT_FLOAT, data_shape);
    a.vec<float>()(0) = 1;

    /** Input tensor should be always dict */
    tensor_dict tensors;
    tensors.push_back(std::make_pair("A:0",a));

    TF_CHECK_OK(
        session->Run(tensors, {"stack_anchor:0"}, {}, &outputs));  

    /** Since there is no other options, anchors are built using tensorflow graph concept */
    /** output will be vector so return 0 indexing */
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout<<duration.count()<<std::endl;
    return outputs[0];

}

tensor_tup LoadGraphAndPredict::LoadBoxes3D(){

    /** TODO:: Just create one Session for all graph */
    /** TODO:: make it to load just .pb file */

    /** Prepare inputs */
    //anchor generation
    auto anchorTensor = LoadAnchor(); //(200,240,2,7) 

    auto anchor = anchorTensor.shaped<float,2>({96000,7});
    Eigen::Tensor<float, 2> anchor_tensor = Eigen::TensorLayoutSwapOp<Eigen::Tensor<float, 2, Eigen::RowMajor>>(anchor);
    Eigen::MatrixXf anchors = Tensor_to_Matrix(anchor_tensor,96000,7);
    
    //model output deltas and probability
    tensor_vec model_output = LoadModel();
    //Delta reshape
    tensorflow::Tensor delta_reshaped(tensorflow::DT_FLOAT,tensorflow::TensorShape({96000,7}));

    if(!delta_reshaped.CopyFrom(model_output[0], tensorflow::TensorShape({96000,7})))
    {
        LOG(ERROR) << "Unsuccessfully reshaped delta tensor [" << model_output[0].DebugString() << "] to [96000,7]";
        /** TODO:: Put the EXIT_FAILURE here */
  
    }
    LOG(INFO) << "Reshaped delta tensor: " << delta_reshaped.DebugString();
    
    auto delta = delta_reshaped.tensor<float,2>();

    Eigen::Tensor<float, 2> delta_tensor = Eigen::TensorLayoutSwapOp<Eigen::Tensor<float, 2, Eigen::RowMajor>>(delta);
    Eigen::MatrixXf deltas = Tensor_to_Matrix(delta_tensor,96000,7);
    
    //Prob reshape
    tensorflow::Tensor prob_reshaped(tensorflow::DT_FLOAT,tensorflow::TensorShape({96000}));

    if(!prob_reshaped.CopyFrom(model_output[1], tensorflow::TensorShape({96000})))
    {
        LOG(ERROR) << "Unsuccessfully reshaped prob tensor [" << model_output[1].DebugString() << "] to [96000]";
         /** TODO:: Put the EXIT_FAILURE here */
  
    }
    LOG(INFO) << "Reshaped prob tensor: " << prob_reshaped.DebugString();

    auto prob_mat = Eigen::Map<Eigen::Matrix<
             float,           /* scalar element type */
             Eigen::Dynamic,  /* num_rows is a run-time value */
             Eigen::Dynamic,  /* num_cols is a run-time value */
             Eigen::RowMajor  /* tensorflow::Tensor is always row-major */>>(
                 prob_reshaped.flat<float>().data(),  /* ptr to data */
                 prob_reshaped.dim_size(0),           /* num_rows */
                 0            /* num_cols */);


    Eigen::VectorXf probs(Eigen::Map<Eigen::VectorXf>(prob_mat.data(), prob_mat.rows()));

    return std::make_tuple(anchors,deltas,probs);
    
}

Eigen::MatrixXf LoadGraphAndPredict::DeltaTo3DBoxes(const Eigen::MatrixXf & deltas, const Eigen::MatrixXf &anchors){
    std::cout<<deltas.rows()<<" "<<deltas.cols()<<std::endl;

     /** Implementation of anchors generation on CPU or GPU might increase the inference time
     * or running the graph on GPU might also increase the inference time */
    Eigen::MatrixXf Boxes3D(96000,7);
    Boxes3D.setZero();
    return Boxes3D;
}

void LoadGraphAndPredict::ApplyNMS(){


auto [anchors,deltas,probs] = LoadBoxes3D();

auto boxes = DeltaTo3DBoxes(deltas,anchors);

// std::vector<float>prob{};
// auto it = std::find_if(std::begin(prob_tensor), std::end(prob_tensor), [](int i){return i > RPN_SCORE_THRESH;});
// while (it != std::end(prob_tensor)) {
//    prob.emplace_back(std::distance(std::begin(prob_tensor), it));
//    it = std::find_if(std::next(it), std::end(prob_tensor), [](int i){return i > RPN_SCORE_THRESH;});
// }

// std::cout<<prob.size()<<std::endl;

}

