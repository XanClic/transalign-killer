#define __TRANSALIGN_KILLER_CU__
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "transalign_killer_thrust.cuh"


int main(int argc, char** argv){
  
  long delta_time;
  struct timeval start_time, end_time;


  if(argc!=2){
    print_help();
    return 1;
  }

  //use the following file: http://idisk.mpi-cbg.de/~steinbac/transalign_sequence.txt.tgz
  //untar it and then use it as input
  std::string file_loc(argv[1]);
  std::cout << "reading input from " << file_loc << "\n";
  std::string seq = get_file_contents(file_loc.c_str());
  if(seq.empty())
    return 1;

  unsigned host_searchedBound=seq.size()/2;

// T *h_ptr = pinned_host_vec.data();
// T *raw_d_ptr = 0;
// cudaHostGetDevicePointer(&raw_d_ptr, h_ptr, 0);
// thrust::device_ptr<T> d_ptr(raw_d_ptr);

  thrust::host_vector<char> host_sequence(host_searchedBound);  
  std::copy(seq.begin(),seq.begin()+host_searchedBound,host_sequence.begin());
  thrust::device_vector<char> device_sequence = host_sequence;
  
  gettimeofday(&start_time, NULL);
  unsigned host_result = thrust::count(device_sequence.begin(),device_sequence.end(),'-');
  
  //total time
  cudaDeviceSynchronize();
  gettimeofday(&end_time, NULL);
  delta_time = (end_time.tv_sec*1000000.+end_time.tv_usec) - (start_time.tv_sec*1000000.+start_time.tv_usec);

  std::cout << "sequence of " << host_searchedBound << " contained " << host_result << " dashes and " << host_searchedBound - host_result << " other characters\n";
  printf(" - %li µs elapsed total\n", delta_time);

  return 0;

}
