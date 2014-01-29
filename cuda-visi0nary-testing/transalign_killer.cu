#include <iostream>
#include <fstream>
#include <string>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>

#define DIMX 10
#define DIMY 10
#define DIMZ 3

#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("<%s>:%i ",__FILE__,__LINE__); printf("[CUDA] Error: %s\n", cudaGetErrorString(error));}}


/** IDEA:
 ** 1. Split sequence into DIMX * DIMY * DIMZ parts
 ** 2. Let each thread sum up the amount of bases on one part
 ** 3. Let each thread sum up the amoutn of bases that were calculated by its predecessors
 ** 4. Pick the thread in which the searched value lies (aka the amount of bases of its predecessors are lower than the searched value AND its amount of bases added to that makes the result greater than the searched value
 ** 5. Search on that part again until we find the desired index
 ** 6. Return the index
 ** 7. ??
 ** 8. PROFIT!
**/

__global__ void sumBases(char *sequence, unsigned *result, unsigned *subSequenceLength, unsigned *searchedBound, size_t *subSequences, size_t sequence_length)
{
    //linearize thread ids
    int threadId = threadIdx.x + threadIdx.y * DIMX + threadIdx.z * DIMX * DIMY;

    subSequences[threadId]=0;


    //count bases in each part of the sequence
    {
        for(size_t i=threadId*(*subSequenceLength); i<(threadId+1)*(*subSequenceLength); i++)
        {
            if(sequence[i]!='-')
            {
                subSequences[threadId]++;
            }
        }
    }

    __syncthreads();


    //sum up the amount of bases which was computed by the "previous" threads (in a linear order)
    size_t cumulatedAmountOfBases=0;
    for(size_t i=0; i<threadId; i++)
    {
        cumulatedAmountOfBases+=subSequences[i];
    }

    __syncthreads();


    //pick the thread that is the last one we look at before we exceed our bound
    if( (cumulatedAmountOfBases < *searchedBound) && (cumulatedAmountOfBases+subSequences[threadId] > *searchedBound))
    {
        //set the result pointer to the first char of the substring
        *result=threadId*(*subSequenceLength);
        //iterate again over the substring
        for(size_t i=threadId*(*subSequenceLength); cumulatedAmountOfBases<*searchedBound; i++)
        {
            if(sequence[i]!='-')
            {
                cumulatedAmountOfBases++;
            }
            //increase the result pointer
            *result=i;
        }
    *result+=1;
    }
}

void print_help(){

  std::cout << "usage: \t transalign_killer <file/to/read/sequence/from>\n";
}


std::string get_file_contents(const char *filename)
{
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  std::string contents("");
  // print file:
  if(in.is_open()){

    while (in.good()) {
      contents.push_back(in.get());
    }
  }
  else
    {
      std::cerr << ">> problem opening file at: " << filename << "\n";
    }

  return contents;

}


int main(int argc, char** argv){
  
  long delta_time;
  struct timeval start_time, end_time;

  //set a starting point
  gettimeofday(&start_time, NULL);

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

  /**convert string to char array**/
  char *host_sequence=new char[seq.size()+1];
  //set the whole sequence to 0
  host_sequence[seq.size()]=0;
  //copy every char
  memcpy(host_sequence, seq.c_str(), seq.size());


  //get integer part for subSequenceLength
  double integerPart;
  modf( seq.size() / (DIMX * DIMY * DIMZ) , &integerPart);
  int iPart = static_cast<int>(integerPart);
  int *host_subSequenceLength = &iPart;


  unsigned *host_searchedBound=(unsigned*) malloc(sizeof(unsigned));
  *host_searchedBound=seq.size()/2;

  //length the part each GPU thread has to deal with
  unsigned *dev_subSequenceLength;
  //pointer for result on device
  unsigned *dev_result;
  //pointer for result on host
  unsigned *host_result=(unsigned*) malloc(sizeof(unsigned));
  //sequence on device
  char *dev_sequence;
  //char array with a slot for each thread on GPU (only a temporary solution for now)
  size_t *dev_subSequences;
  unsigned *dev_searchedBound;


  /**start GPU stuff**/
  dim3 block(DIMX, DIMY, DIMZ);

  CUDA_CHECK(cudaMalloc((void**)&dev_result, sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc((void**)&dev_subSequenceLength, sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc((void**)&dev_searchedBound, sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc((void**)&dev_sequence, seq.size()*sizeof(char)));
  CUDA_CHECK(cudaMalloc((void**)&dev_subSequences, DIMX * DIMY * DIMZ * sizeof(size_t)));


  //this is where things start to become incredibly slow
  CUDA_CHECK(cudaMemcpy(dev_sequence, host_sequence, seq.size()*sizeof(char), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_subSequenceLength, host_subSequenceLength, sizeof(unsigned), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_searchedBound, host_searchedBound, sizeof(unsigned), cudaMemcpyHostToDevice));

  gettimeofday(&end_time, NULL);
  long bw1_time = (end_time.tv_sec*1000000+end_time.tv_usec) - (start_time.tv_sec*1000000+start_time.tv_usec);

  sumBases<<<1,block>>>(dev_sequence, dev_result, dev_subSequenceLength, dev_searchedBound, dev_subSequences, seq.size());

  gettimeofday(&end_time, NULL);
  long gpu_time = (end_time.tv_sec*1000000+end_time.tv_usec) - (start_time.tv_sec*1000000+start_time.tv_usec);
  
  CUDA_CHECK(cudaMemcpy(host_result, dev_result, sizeof(unsigned), cudaMemcpyDeviceToHost));
  
  gettimeofday(&end_time, NULL);
  long bw2_time = (end_time.tv_sec*1000000+end_time.tv_usec) - (start_time.tv_sec*1000000+start_time.tv_usec);

  printf("Result: %u \n", *host_result);

  CUDA_CHECK(cudaFree(dev_sequence));
  CUDA_CHECK(cudaFree(dev_result));
  CUDA_CHECK(cudaFree(dev_subSequenceLength));
  CUDA_CHECK(cudaFree(dev_subSequences));
  CUDA_CHECK(cudaFree(dev_searchedBound));
  free(host_result);

  //total time
  gettimeofday(&end_time, NULL);
  delta_time = (end_time.tv_sec*1000000+end_time.tv_usec) - (start_time.tv_sec*1000000+start_time.tv_usec);


  printf(" - %li µs elapsed total\n", delta_time);
  printf(" - %li µs on bandwidth forth\n", bw1_time);
  printf(" - %li µs on GPU\n", gpu_time - bw1_time);
  printf(" - %li µs on bandwidth back\n", bw2_time - gpu_time);
  printf(" - %li µs on CPU\n", delta_time - bw2_time);

  return 0;

}
