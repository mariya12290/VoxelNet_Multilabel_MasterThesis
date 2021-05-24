/** Eigen header files */
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>


template<typename T>
using  MatrixType = Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor>;

template<typename Scalar,int rank, typename sizeType>
auto Tensor_to_Matrix(const Eigen::Tensor<Scalar,rank> &tensor,const sizeType rows,const sizeType cols)
{
    return Eigen::Map<const MatrixType<Scalar>> (tensor.data(), rows,cols);
}


namespace Eigen {
  template < typename T >
  decltype(auto) TensorLayoutSwap(T&& t)
  {
    return Eigen::TensorLayoutSwapOp<typename std::remove_reference<T>::type>(t);
  }
}