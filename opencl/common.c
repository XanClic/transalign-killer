#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <CL/cl.h>

#include "common.h"

// See common.h for documentation
cl_device_id select_device(int argc, char *argv[])
{
    struct dev_info
    {
        struct dev_info *next;
        cl_device_id dev;
        cl_device_type type;
        int cus;
        char *driver_ver;
    } *di_list = NULL, **di_ptr = &di_list;


    int plat = -1, dev = -1;

    for (int i = 1; i < argc; i++)
    {
        if (!strncmp(argv[i], "--cldev=", sizeof("--cldev=") - 1))
        {
            sscanf(argv[i], "--cldev=%i.%i", &plat, &dev);
            assert((plat >= 0) && (dev >= 0));
            break;
        }
    }


    cl_device_id selected = NULL;

    if ((plat >= 0) || (dev >= 0))
    {
        cl_uint plat_avail;
        cl_platform_id plats[plat + 1];
        clGetPlatformIDs(plat + 1, plats, &plat_avail);
        assert(plat < (int)plat_avail);

        cl_uint dev_avail;
        cl_device_id devs[dev + 1];
        clGetDeviceIDs(plats[plat], CL_DEVICE_TYPE_ALL, dev + 1, devs, &dev_avail);
        assert(dev < (int)dev_avail);

        selected = devs[dev];
    }
    else
    {
        cl_uint plat_avail;
        clGetPlatformIDs(0, NULL, &plat_avail);

        printf("%u platforms available:\n", plat_avail);

        cl_platform_id plats[plat_avail];
        clGetPlatformIDs(plat_avail, plats, NULL);

        for (int i = 0; i< (int)plat_avail; i++)
        {
            char name[256];

            memset(name, 0, 256);
            clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, 255, name, NULL);

            printf("  [%i] %s\n", i, name);

            cl_uint avail;
            clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, 0, NULL, &avail);

            printf("    %u devices available:\n", avail);

            cl_device_id devs[avail];
            clGetDeviceIDs(plats[i], CL_DEVICE_TYPE_ALL, avail, devs, NULL);

            for (int j = 0; j < (int)avail; j++)
            {
                memset(name, 0, 256);
                clGetDeviceInfo(devs[j], CL_DEVICE_NAME, 255, name, NULL);

                printf("      [%i] %s\n", j, name);

                cl_device_type type;
                clGetDeviceInfo(devs[j], CL_DEVICE_TYPE, sizeof(type), &type, NULL);

                cl_uint cus;
                clGetDeviceInfo(devs[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);

                switch (type)
                {
                    case CL_DEVICE_TYPE_CPU: printf("        CPU (%u CUs)\n", cus); break;
                    case CL_DEVICE_TYPE_GPU: printf("        GPU (%u CUs)\n", cus); break;
                    case CL_DEVICE_TYPE_ACCELERATOR: printf("        Accelerator (%u CUs)\n", cus); break;
                    default: printf("        Unknown (%u CUs)\n", cus);
                }

                cl_ulong memsz;
                clGetDeviceInfo(devs[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memsz), &memsz, NULL);

                cl_ulong lmemsz;
                clGetDeviceInfo(devs[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lmemsz), &lmemsz, NULL);

                cl_device_local_mem_type lmemtype;
                clGetDeviceInfo(devs[j], CL_DEVICE_LOCAL_MEM_TYPE, sizeof(lmemtype), &lmemtype, NULL);

                printf("        %iM global memory; %ik local memory (%s)\n", (int)(memsz >> 20), (int)(lmemsz >> 10), lmemtype == CL_LOCAL ? "dedicated" : "part of global");


                struct dev_info *di = malloc(sizeof(*di));
                di->next = NULL;
                di->dev = devs[j];
                di->type = type;
                di->cus = cus;
                di->driver_ver = calloc(256, 1);
                clGetDeviceInfo(devs[j], CL_DRIVER_VERSION, 255, di->driver_ver, NULL);

                *di_ptr = di;
                di_ptr = &di->next;
            }
        }


        if (!di_ptr)
            return NULL;


        cl_device_type best_type = CL_DEVICE_TYPE_CPU;

        for (struct dev_info *di = di_list; di; di = di->next)
        {
            if (di->type == CL_DEVICE_TYPE_ACCELERATOR)
                best_type = CL_DEVICE_TYPE_ACCELERATOR;
            else if ((di->type == CL_DEVICE_TYPE_GPU) && (best_type == CL_DEVICE_TYPE_CPU))
                best_type = CL_DEVICE_TYPE_GPU;
        }

        int highest_cus = 0;

        for (struct dev_info *di = di_list; di; di = di->next)
            if (di->cus > highest_cus)
                highest_cus = di->cus;

        for (struct dev_info **di = &di_list; *di;)
        {
            if (((*di)->type != best_type) || ((*di)->cus < highest_cus))
            {
                struct dev_info *odi = *di;
                *di = odi->next;
                free(odi->driver_ver);
                free(odi);
            }
            else
                di = &(*di)->next;
        }

        const char *best_ver = NULL;

        for (struct dev_info *di = di_list; di; di = di->next)
        {
            if (!best_ver || (strcmp(di->driver_ver, best_ver) > 0))
            {
                best_ver = di->driver_ver;
                selected = di->dev;
            }
        }

        for (struct dev_info *di = di_list; di;)
        {
            struct dev_info *odi = di;
            di = di->next;
            free(odi->driver_ver);
            free(odi);
        }
    }


    char vendor[256] = { 0 }, name[256] = { 0 }, drv_ver[256] = { 0 };
    clGetDeviceInfo(selected, CL_DEVICE_VENDOR, 255, vendor, NULL);
    clGetDeviceInfo(selected, CL_DEVICE_NAME, 255, name, NULL);
    clGetDeviceInfo(selected, CL_DRIVER_VERSION, 255, drv_ver, NULL);
    printf("Selected device %s %s (driver %s)\n", vendor, name, drv_ver);


    return selected;
}
