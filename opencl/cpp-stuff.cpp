#include <chrono>


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
