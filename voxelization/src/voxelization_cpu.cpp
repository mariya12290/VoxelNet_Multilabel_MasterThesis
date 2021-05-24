 /** Standard header files */
#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <experimental/filesystem>
#include <set>
#include <functional>

/** Internal Header files */
#include "voxelization_cpu.hpp"


PreProcessing::PreProcessing(const std::string& bin_path):bin_file{bin_path} {
/** Do something */
ReadPointCloud();

}

void PreProcessing::ReadPointCloud(){

    std::fstream input(bin_file.c_str(), std::ios::binary | std::ios::in );
    if(!input.good()){
		std::cerr << "Could not read point cloud files: " << bin_file << std::endl;
		exit(EXIT_FAILURE);
	}
    input.seekg(0, std::ios::beg);
	int i;
   
	for (i=0; input.good() && !input.eof(); i++) {
		pcl::PointXYZI point;
		input.read((char *) &point.x, 3*sizeof(float));
		input.read((char *) &point.intensity, sizeof(float));
		points.push_back(point); 
        /** can not use emplace back or move, since class pcl::PointCloud<pcl::PointXYZI> is custom data structure
         * for better optimization have a look into below line number 66 */
	}
    input.close();
    number_of_points = points.size();
}

Eigen::MatrixXi PreProcessing::EigenRowSort(Eigen::MatrixXi& mat){
    std::vector<std::vector<int>>vec;
        for (int64_t i = 0; i < mat.rows(); ++i)
        {
            Eigen::VectorXi v = mat.row(i);
            std::vector<int>arr(v.data(),v.data()+v.size());
            vec.push_back(arr);
        }
    /** TODO:: find an optimal solution to remove copying from set to vector */
    std::set<std::vector<int>>s (vec.begin(),vec.end());
    vec.assign(s.begin(),s.end());
   
  
    Eigen::MatrixXi mat_(vec.size(),3);
        for (int64_t i = 0; i < vec.size(); ++i)
        {
            Eigen::VectorXi v = Eigen::VectorXi::Map(vec[i].data(),vec[i].size());
            mat_.row(i) = v;
        }
        return mat_;
      
}

tensor_dict PreProcessing::PointCloudProcess(){
    /** it is redundant copy operation so instead of reading the point cloud to points and again copying points to point_cloud(below)
     * directly read the file into point_cloud */
    Eigen::MatrixXf point_cloud(number_of_points,4);  //(None,4)
    for(auto i{0};i<number_of_points;++i){
       point_cloud(i,0) = static_cast<float>(points[i].x);
       
       point_cloud(i,1) = static_cast<float>(points[i].y);
       point_cloud(i,2) = static_cast<float>(points[i].z);
       point_cloud(i,3) = static_cast<float>(points[i].intensity);
   }

   /** TODO:: Shuffle the point cloud here */
   
   std::vector<int> ind{0,1,2};
   auto shifted_cord = point_cloud(Eigen::all,ind); //(None,3)

   Eigen::Vector3f v = Eigen::Vector3f(0,20.,3);
   
   shifted_cord = shifted_cord.rowwise()+v.transpose();

   shifted_cord = shifted_cord.rowwise().reverse().eval();

   Eigen::Vector3f voxel_size = Eigen::Vector3f(0.4,0.2,0.2);
   auto voxel_index = shifted_cord.array().rowwise()/voxel_size.transpose().array();  //(None,3)

   auto voxel_index_1 = voxel_index.floor();

   Eigen::MatrixXi voxel_index_2(voxel_index_1.cast<int>());
   
   std::array<int,3> grid_size{10,200,240};
   
   std::array<std::vector<int>,3>voxel_index_col;
    /** TODO:: Try to remove for loop at least 1 and push_back as well*/
    for(auto count{0};count <3;count++){
        for(auto i{0};i<voxel_index_2.rows();++i){
            if(voxel_index_2(i,count)>= 0 && voxel_index_2(i,count)< grid_size[count]){
                voxel_index_col[count].push_back(1);
            }else{
                voxel_index_col[count].push_back(0);
            }
        }
    }
    std::vector<bool>index;
    /** TODO:: Try to remove the push_back instead use move if possible */
    for(auto i{0};i<voxel_index_col[0].size();++i){
        if(voxel_index_col[0][i] && voxel_index_col[1][i]&&voxel_index_col[2][i]){
            index.push_back(1);
        }else{
            index.push_back(0);
        }
    }

    std::vector<int>keep_rows;
    /** TODO:: Try to remove the push_back instead use move if possible */
    for(auto i{0};i<index.size();i++){
        if(index[i] ==1){
            keep_rows.push_back(i);
        }
    }

    Eigen::MatrixXi voxel_index_3 = voxel_index_2(keep_rows,Eigen::all); //(None,3)
    Eigen::MatrixXf point_cloud_1 = point_cloud(keep_rows,Eigen::all); //(None,4)


    Eigen::MatrixXi coordinate(voxel_index_3);
    Eigen::MatrixXi coordinate_buffer =  EigenRowSort(coordinate);

    auto K = coordinate_buffer.rows();

    Eigen::VectorXi number_buffer(K);
    Eigen::Tensor<float,3> feature_buffer(K,45,7);
    number_buffer.setZero();
    feature_buffer.setZero();

    std::unordered_map<Key,int,KeyHash> index_buffer;
    for (auto i{0};i<coordinate_buffer.rows();++i){
        Eigen::VectorXi rows = coordinate_buffer.row(i);
        index_buffer[std::make_tuple(rows(0),rows(1),rows(2))] = i;
    }

    for(auto i{0};i<point_cloud_1.rows();++i){
        Eigen:: VectorXi rows = voxel_index_3.row(i);
        auto index_1 = index_buffer[std::make_tuple(rows(0),rows(1),rows(2))];
        auto number = number_buffer(index_1);
        if (number< 45)
        {
            Eigen::VectorXf row = point_cloud_1.row(i);
            feature_buffer(index_1,number,0) = row(2);
            feature_buffer(index_1,number,1) = row(1)-20.0;
            feature_buffer(index_1,number,2) = row(0) - 3.0;
            feature_buffer(index_1,number,3) = row(3);
            number_buffer(index_1) +=1;
        }
    }

    /** For loop is too expensive find optimal solution to this in a vectorised form
     * Since we are dealing with Tensors of 3D, there are not enough support or API to work 
     * on 3D tensors, So used for loop
    */

    Eigen::MatrixXf feature_sum(K,3);

    float a{0};
    float b{0};
    float c{0};
    /** TODO:: Try to remove for loop at least 1*/
    for(int i{0};i<K;++i){
        for(int j{0};j<45;++j){
            a += feature_buffer(i,j,0);
            b  += feature_buffer(i,j,1);
            c +=  feature_buffer(i,j,2);
    }
    

   feature_sum(i,0) = a/number_buffer(i);
   feature_sum(i,1) = b/number_buffer(i);
   feature_sum(i,2) = c/number_buffer(i);
   a = 0;
   b =0;
   c =0;
    }

    Eigen::Tensor<float,3> feature_minus(K,45,3);
    /** TODO:: Try to remove for loop at least 1*/
    for(auto i{0};i<K;++i){
        for(auto j{0};j<45;++j){
            feature_minus(i,j,0) = feature_buffer(i,j,0) - feature_sum(i,0);
            feature_minus(i,j,1) = feature_buffer(i,j,1) - feature_sum(i,1);
            feature_minus(i,j,2) = feature_buffer(i,j,2) - feature_sum(i,2);
        }
    }
    /** TODO:: Try to remove for loop at least 1*/
    for(auto i{0};i<K;++i){
        for(auto j{0};j<45;++j){
            feature_buffer(i,j,4) = feature_minus(i,j,0);
            feature_buffer(i,j,5) = feature_minus(i,j,1);
            feature_buffer(i,j,6) = feature_minus(i,j,2);
        }
    }
 /** TODO:: use just for loop */
    tensorflow::Tensor feature_tensor{tensorflow::DataType::DT_FLOAT, tensorflow::TensorShape({K,45,7})};
        auto featureMapped = feature_tensor.tensor<float, 3>();
        /** TODO:: Try to remove for loop at least 1*/
        for(int i=0; i<K; i++){
            for(int j{0};j<45;++j){
                featureMapped(i,j,0) = feature_buffer(i,j,0);
                featureMapped(i,j,1) = feature_buffer(i,j,1);
                featureMapped(i,j,2) = feature_buffer(i,j,2);
                featureMapped(i,j,3) = feature_buffer(i,j,3);
                featureMapped(i,j,4) = feature_buffer(i,j,4);
                featureMapped(i,j,5) = feature_buffer(i,j,5);
                featureMapped(i,j,6) = feature_buffer(i,j,6);

            }
        }
 
    tensorflow::Tensor number_tensor{tensorflow::DataType::DT_INT64,tensorflow::TensorShape{K}};
    auto numberMapped = number_tensor.tensor<int64_t,1>();
    for(auto i{0};i<K;++i){
        numberMapped(i) = number_buffer(i);
    }

    tensorflow::Tensor coordinate_tensor{tensorflow::DataType::DT_INT64,tensorflow::TensorShape{K,4}};
    auto coordinateMapped = coordinate_tensor.tensor<int64_t,2>();
    for(auto i{0};i<K;++i){
        Eigen::VectorXi row =  coordinate_buffer.row(i);
        coordinateMapped(i,0) = 0;
        coordinateMapped(i,1) = row(0);
        coordinateMapped(i,2) = row(1);
        coordinateMapped(i,3) = row(1);
        
    }
    tensor_dict tensor{};

    tensorflow::Tensor phase_tensor{tensorflow::DataType::DT_BOOL,tensorflow::TensorShape{1}};
    phase_tensor.vec<bool>()(0) = false;

    /** Here depending upon GPU available change the number here 
     * for an example gpu_0 for gpu 0 */
    tensor.push_back(std::make_pair("phase", phase_tensor));
    tensor.push_back(std::make_pair("gpu_1/coordinate:0",coordinate_tensor)); //(K,3)
    tensor.push_back(std::make_pair("gpu_1/number:0",number_tensor)); //(K)
    tensor.push_back(std::make_pair("gpu_1/feature:0",feature_tensor)); //(K,45,7)

   return std::move(tensor);  /** try NRVO (named return value optimization) instead of move*/
   // currently taking around 350 ms 
}


