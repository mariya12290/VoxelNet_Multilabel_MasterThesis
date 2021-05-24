/** Internal header files */
#include "load_graphs.hpp"

int main(){
std::string bin_file = "/home/surendra/data/kitti/cropped_kitti/validation/velodyne/001101.bin";

std::unique_ptr<LoadGraphAndPredict> model = std::make_unique<LoadGraphAndPredict>(bin_file);

auto result = model->LoadModel();
return 0;

}