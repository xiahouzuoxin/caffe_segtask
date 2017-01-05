
#ifndef CAFFE_TVG_COMMON_UTILS_HPP
#define CAFFE_TVG_COMMON_UTILS_HPP

#include <string>
#include "caffe/blob.hpp"

namespace tvg {

  namespace CommonUtils {


    template<typename Dtype>
    void read_into_the_diagonal(const std::string & source, caffe::Blob<Dtype> & blob) {

      const int height = blob.height();
      Dtype * data = blob.mutable_cpu_data();
      caffe::caffe_set(blob.count(), Dtype(0.), data);

      std::stringstream iss;
      iss.clear();
      iss << source;
      std::string token;

	  int i = 0;
      while (std::getline(iss, token, ' ')) {
        data[i * height + i] = std::stof(token);
		i++;
      }
	  for ( ; i < height; ++i) {  // remain value
	    data[i * height + i] = std::stof(token);
	  }
    }

  }  // end namespace CommonUtils
}  // end namespace tvg


#endif //CAFFE_TVG_COMMON_UTILS_HPP

