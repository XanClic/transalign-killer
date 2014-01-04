#ifndef COMMON_H
#define COMMON_H

#include <CL/cl.h>


/**
 * This function merely tries to select the best OpenCL device available. If not
 * specified through the command line (via --cldev=x.y with x being the platform
 * and y the device), it will first prefer accelerator devices, then GPUs and
 * use CPUs last. Then it will prefer the device with the highest number of CUs
 * and finally the one with the newest driver version (strcmp() > 0).
 *
 * Selecting an OpenCL device can be a PITA (especially with multiple platforms),
 * so it's nice to have such a blobby function which does the job.
 */
cl_device_id select_device(int argc, char *argv[]);

#endif
