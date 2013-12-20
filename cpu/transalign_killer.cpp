//to compile with gcc, do: g++ -O3 -march=native -std=c++11 __FILE__
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

template <const char _cmp>
unsigned count_default(unsigned _lower, const unsigned& _upper, const char* _refrow ){
  unsigned value = 0;

  while (_lower < _upper)
    {
      if (_refrow[value++] != _cmp)
    		++_lower;
    }

  return value;
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

template <const char _cmp,  typename IndexT,typename IterT>
IndexT vec_count_no_branch(IterT _begin, IterT _end){
  IndexT value = 0;
  asm("#start loop vec_count_no_branch");
  for (; _begin!=_end;){
    value += (*_begin++ == _cmp);
  }
  asm("#end loop vec_count_no_branch");
  return value;
}

template <const char _cmp>
unsigned count_with_exceptions_recur_my_count_no_branch(unsigned _lower, const unsigned& _upper, const char* _refrow, const unsigned& _start ){

  unsigned value = _start;
  // if(!(_lower < _upper))
  //   return value;
  
  unsigned initial_distance = _upper - _lower;

  //icpc and clang vectorizes std::count, surprisingly gcc does not
#if !(defined __clang__ || defined __INTEL_COMPILER)
  const char* refrow = (const char*)__builtin_assume_aligned(_refrow,8);
#else
  const char* refrow = _refrow;
#endif

  unsigned num_dashes = vec_count_no_branch<_cmp,unsigned>(&refrow[value],&refrow[value+initial_distance]);
  //std::count(,_cmp);
  value += initial_distance;
  _lower += initial_distance - num_dashes;

  if(_lower < _upper){
    return count_with_exceptions_recur_my_count_no_branch<_cmp>(_lower, 
					     _upper, 
					     refrow, 
					     value);
  }
  else{
    return value;
  }
  
}

int main(int argc, char** argv){

  if(argc!=2){
    print_help();
    return 1;
  }

  //use the following file: http://idisk.mpi-cbg.de/~steinbac/transalign_sequence.txt.tgz
  //untar it and then use it as input
  std::string file_loc(argv[1]);
  std::cout << "reading input from " << file_loc << "\n";
  std::string sequence = get_file_contents(file_loc.c_str());
  if(sequence.empty())
    return 1;

  unsigned lower_letter_index = 10;
  unsigned upper_letter_index = sequence.size()-10;
  std::cout << "lower index: " << lower_letter_index << ", upper index: " << upper_letter_index << "\n";
  static const char search_character = '-';
  
  auto t_start = std::chrono::high_resolution_clock::now();
  unsigned count_wo_dashes = count_default<search_character>(lower_letter_index,upper_letter_index,sequence.c_str());
  auto t_end = std::chrono::high_resolution_clock::now();
  double delta_default = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
  std::cout << "default method     \t" << delta_default  <<" ms, cnt = "<< count_wo_dashes<<"\n";

  return 0;

}
