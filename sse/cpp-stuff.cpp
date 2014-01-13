#include <chrono>


/**
 * These functions are here because std::chrono is an admittedly nice part of
 * the C++ STL. Using gettimeofday() is nice as well, but breaks on Windows.
 * Therefore, here are two functions exporting a tiny part of the std::chrono
 * functionality to the C part of the program.
 */


static std::chrono::time_point<std::chrono::high_resolution_clock> start;

extern "C" void clock_start(void)
{
    start = std::chrono::high_resolution_clock::now();
}


extern "C" long clock_delta(void)
{
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}
