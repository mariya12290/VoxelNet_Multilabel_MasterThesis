#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

using tensor_dict = std::vector<std::pair<std::string,tensorflow::Tensor>>;

tensorflow::Status LoadGraph(tensorflow::Session *sess, std::string graph_fn,
                             std::string checkpoint_fn = "") {
  tensorflow::Status status;

  // Read in the protobuf graph we exported
  tensorflow::MetaGraphDef graph_def;
  status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
  if (status != tensorflow::Status::OK()) return status;

  // create the graph
  status = sess->Create(graph_def.graph_def());
  if (status != tensorflow::Status::OK()) return status;

  // restore model from checkpoint, if checkpoint is given
  if (checkpoint_fn != "") {
    tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING,
                                            tensorflow::TensorShape());
    checkpointPathTensor.scalar<tensorflow::tstring>()() = checkpoint_fn;

    tensor_dict feed_dict = {
        {graph_def.saver_def().filename_tensor_name(), checkpointPathTensor}};
    status = sess->Run(feed_dict, {}, {graph_def.saver_def().restore_op_name()},
                       nullptr);
    if (status != tensorflow::Status::OK()) return status;
  } else {
    status = sess->Run({}, {}, {"init"}, nullptr);
    if (status != tensorflow::Status::OK()) return status;
  }

  return tensorflow::Status::OK();
}
