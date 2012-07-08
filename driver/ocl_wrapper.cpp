#include <ocl_wrapper.h>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <assert.h>
#include <cmath>
#include <gsl/gsl_statistics_double.h>

using namespace std;

cl_ulong local_mem_size;

ocl_test::ocl_test() : logfile("openclbenchmark.log"), alloc_size(64*1024*1024)
{
  register_tests();
}

void ocl_test::run_tests()
{
  for (int i = 0; i < int(ocl_test_funcs.size()); i++)
  {
		logfile << "test #" << i << endl;
    ocl_test_funcs[i](*this);
  }
}

void ocl_test::get_max_sizes()
{
  clGetDeviceInfo(devices[cur_dev_num], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, NULL);
  clGetDeviceInfo(devices[cur_dev_num], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dimensions), &max_work_item_dimensions, NULL);
  clGetDeviceInfo(devices[cur_dev_num], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), max_work_item_sizes, NULL);
  clGetDeviceInfo(devices[cur_dev_num], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
  assert(max_work_item_dimensions >= 3);
}

void ocl_test::run_tests_on_all()
{
  cl_int errnum;
  cl_uint numPlatforms = 0x10000;
  cl_platform_id platforms[0x10000];
  
  clGetPlatformIDs(numPlatforms, platforms, &numPlatforms);

  logfile << "Available platforms: " << endl;

  for (int i = 0; i < int(numPlatforms); i++)
  {
    char pbuf[100];
    
    clGetPlatformInfo(
      platforms[i],
      CL_PLATFORM_VENDOR,
      sizeof(pbuf),
      pbuf,
      NULL);
    logfile << pbuf << endl;
  }

  for (int i = 0; i < int(numPlatforms); i++)
  {
    char pbuf[0x10000];
    
    clGetPlatformInfo(
      platforms[i],
      CL_PLATFORM_VENDOR,
      sizeof(pbuf),
      pbuf,
      NULL);
    
    logfile << "Testing: " << pbuf << endl;
  
    cl_context_properties cps[3] = 
    {
      CL_CONTEXT_PLATFORM, 
      (cl_context_properties)platforms[i], 
      0
    };

    devices.resize(0x100000);
    
    cl_uint num = 0;
    errnum = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, devices.size(), &devices[0], &num);
    geterr(errnum, __LINE__, "ocl_wrapper.cpp");
    devices.resize(num);
    
    for (int j = 0; j < int(devices.size()); j++)
    {
      cl_device_id cur_dev = devices[j];
      cur_dev_num = j;
      get_max_sizes();
      
      stringstream ss;
      
      clGetDeviceInfo(cur_dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);

      size_t ret_size;
      clGetDeviceInfo(cur_dev, CL_DEVICE_NAME, sizeof(pbuf), pbuf, &ret_size);
      ss << string(pbuf, pbuf+ret_size-1);
      clGetDeviceInfo(cur_dev, CL_DEVICE_VERSION, sizeof(pbuf), pbuf, &ret_size);
      ss << "<#>" << string(pbuf, pbuf+ret_size-1);
      ss << "<#>" << i;
      
      dev_name = ss.str();
      
      logfile << "Testing: " << dev_name << endl;
      
      context = clCreateContext(cps, 1, &cur_dev, NULL, NULL, &errnum);
      geterr(errnum, __LINE__, "ocl_wrapper.cpp");
      
			logfile << "context :" << context << endl;
			
      command_queue = clCreateCommandQueue(context, cur_dev, CL_QUEUE_PROFILING_ENABLE, &errnum);
      geterr(errnum, __LINE__, "ocl_wrapper.cpp");
      logfile << "cmdqueue :" << command_queue << endl;
      alloc_memory();
      
      run_tests();
      
      free_memory();
      
      clReleaseCommandQueue(command_queue);
      clReleaseContext(context);
    }
  }
}

void ocl_test::register_ocl_test(test_func f, std::string name)
{
  ocl_test_funcs_by_name[name] = f;
  ocl_test_funcs.push_back(f);
}

void ocl_test::register_gold_test(test_func f, std::string name)
{
  gold_test_funcs_by_name[name] = f;
  gold_test_funcs.push_back(f);
}

cl_program ocl_test::ocl_load_src(const char* src)
{
  cl_int errnum = 0;
  size_t len = strlen(src);
  
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&src, &len, &errnum);
  geterr(errnum, __LINE__, "ocl_wrapper.cpp");
    
  stringstream ss;
  ss << "-Dlocal_mem_size=" << local_mem_size;
  
  errnum = clBuildProgram(program, 1, &devices[cur_dev_num], ss.str().c_str(), NULL, NULL);
  
  char buf[100000];
  size_t rlen = 0;
  clGetProgramBuildInfo(program, devices[cur_dev_num], CL_PROGRAM_BUILD_LOG, sizeof(buf), buf, &rlen);
  
  buf[rlen] = 0;

  logfile << buf << endl;
  
  geterr(errnum, __LINE__, "ocl_wrapper.cpp");
  
  return program;
}

bool ocl_test::interesting_number(long num, std::string name)
{
  if (kernel_flags[name]&KERNEL_FLAG_ALL_LOCAL_SIZES)
  {
    return true;
  }
  
  if (num <= 16)
  {
    return true;
  }
  
  if (abs(long(pow(2, int(round(log2(num)))))-num) < 6)
  {
    return true;
  }
  
  int pnum = 0;
  
  long m = num;
  
  while (m != 1)
  {
    for (int i = 2; i <= num; i++)
    {
      while (m%i == 0)
      {
        m /= i;
        pnum++;
      }
    }
  }
  
  if (kernel_flags[name]&KERNEL_FLAG_MORE_LOCAL_SIZES)
  {
    pnum++;
  }
  
  if ((pnum > 4 and num < 255) or pnum > 5)
  {
    return true;
  }
  
  return false;
}

void ocl_test::launch_kernel(cl_kernel kernel, const char* name)
{
  logfile << "Testing kernel: " << name << endl;
  
  int global_max_x = alloc_size/sizeof(int);
  int global_max_y = alloc_size/sizeof(int);
  int global_max_z = alloc_size/sizeof(int);
  
  int local_max_x = max_work_item_sizes[0];
  int local_max_y = max_work_item_sizes[1];
  int local_max_z = max_work_item_sizes[2];
  
  if (max_global_size[name][0] == -1)
  {
    global_max_x = local_max_x*max_compute_units;
  }
  
  if (max_global_size[name][1] == -1)
  {
    global_max_y = local_max_y*max_compute_units;
  }
  
  if (max_global_size[name][2] == -1)
  {
    global_max_z = local_max_z*max_compute_units;
  }
  
  if (max_global_size[name][0] > 0)
  {
    global_max_x = max_global_size[name][0];
  }
  
  if (max_global_size[name][1] > 0)
  {
    global_max_y = max_global_size[name][1];
  }
  
  if (max_global_size[name][2] > 0)
  {
    global_max_z = max_global_size[name][2];
  }
  
  if (max_local_size[name][0] > 0)
  {
    local_max_x = max_local_size[name][0];
  }
  
  if (max_local_size[name][1] > 0)
  {
    local_max_y = max_local_size[name][1];
  }
  
  if (max_local_size[name][2] > 0)
  {
    local_max_z = max_local_size[name][2];
  }
  
  size_t work_group_computed_size;
  
  clGetKernelWorkGroupInfo(kernel, devices[cur_dev_num], CL_KERNEL_WORK_GROUP_SIZE, sizeof(work_group_computed_size), &work_group_computed_size, NULL);
  
  logfile << "work_group_computed_size: " << work_group_computed_size << endl;
  
  //local_max_x = local_max_y = local_max_z = 1;
  
  long lmx = max(min_local_size[name][0], 1);
  long lmy = max(min_local_size[name][1], 1);
  long lmz = max(min_local_size[name][2], 1);
  
  ///fallback code in case the minimum local size is too big
  if (lmx*lmy*lmz > work_group_computed_size or lmx*lmy*lmz > max_work_group_size)
  {
    lmx = lmy = lmz = 1;
  }
  
  if (lmx > local_max_x)
  {
    lmx = 1;
  }
  
  if (lmy > local_max_y)
  {
    lmy = 1;
  }
  
  if (lmz > local_max_z)
  {
    lmz = 1;
  }
  
  for (long lx = lmx; lx <= local_max_x; lx++)
  for (long ly = lmy; ly <= local_max_y; ly++)
  for (long lz = lmz; lz <= local_max_z; lz++)
  if (lx*ly*lz <= max_work_group_size and lx*ly*lz <= work_group_computed_size)
  if (interesting_number(lx*ly*lz, name))
  {
    long local_group_size = lx*ly*lz;
    
    int ratio = 32 / local_group_size;
    
    if (ratio == 0)
    {
      ratio = 1;
    }
    
    long global_max2_x = max(global_max_x / ratio, 1);
    long global_max2_y = max(global_max_y / ratio, 1);
    long global_max2_z = max(global_max_z / ratio, 1);
    
    int rnum = 3;
    
    if (global_max2_x > 1 and global_max2_y == 1 and global_max2_z == 1)
    {
      rnum = 16;
    }

    if (global_max2_x > 1 and global_max2_y > 1 and global_max2_z == 1)
    {
      rnum = 4;
    }

    long gx_increment = ((global_max2_x/rnum + lx - 1) / lx)*lx;
    long gy_increment = ((global_max2_y/rnum + ly - 1) / ly)*ly;
    long gz_increment = ((global_max2_z/rnum + lz - 1) / lz)*lz;
    
    if (gx_increment == 0)
    {
      gx_increment = lx;
    }
    
    if (gy_increment == 0)
    {
      gy_increment = ly;
    }
    
    if (gz_increment == 0)
    {
      gz_increment = lz;
    }
    
    for (long gx = lx; gx <= global_max2_x; gx += gx_increment)
    for (long gy = ly; gy <= global_max2_y; gy += gy_increment)
    for (long gz = lz; gz <= global_max2_z; gz += gz_increment)
    if (gx*gy*gz <= (alloc_size/sizeof(int)))
    {
      test_iden iden;
      iden.dev_name = dev_name;
      iden.test_name = name;
      iden.local_size[0] = lx;
      iden.local_size[1] = ly;
      iden.local_size[2] = lz;
      iden.global_size[0] = gx;
      iden.global_size[1] = gy;
      iden.global_size[2] = gz;
      
      logfile << "test config : " << lx << ":" << ly << ":" << lz << "\t" << gx << ":" << gy << ":" << gz << endl;
      
      test_configuration(kernel, iden);
    }
  }
}

double ocl_test::event_to_time(cl_event event)
{
  cl_ulong q_t, end_t;
  
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(q_t), &q_t, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end_t), &end_t, NULL);

  double res = double(end_t-q_t) / 1000.0;
  
  clReleaseEvent(event);
  
  return res;
}

static double sqr(double x)
{
  return x*x;
}

void ocl_test::test_configuration(cl_kernel kernel, test_iden ident)
{
  result_summary summary_final;
  
  summary_final.JB = 1E40;
  
  for (int w = 0; w < 5; w++)
  {
    cl_int errnum;
    cl_event event;
    
    ///warmup:
    errnum = clEnqueueNDRangeKernel(command_queue, kernel, 3, NULL, &ident.global_size[0], &ident.local_size[0], 0, NULL, &event);
    geterr(errnum, __LINE__, "ocl_wrapper.cpp");
    clFinish(command_queue);
    
    double warmup_time = event_to_time(event);
    
    int rounds = (double(4*1000*1000) / warmup_time);
    
    if (rounds > 256)
    {
      rounds = 256;
    }
    
    logfile << rounds << " rounds, wtime: " << warmup_time << "us" << endl;
    
    assert(rounds > 0);
    
    vector<cl_event> events;
    
    for (int i = 0; i < rounds; i++)
    {
      cl_event event;
      errnum = clEnqueueNDRangeKernel(command_queue, kernel, 3, NULL, &ident.global_size[0], &ident.local_size[0], 0, NULL, &event);
      events.push_back(event);
      geterr(errnum, __LINE__, "ocl_wrapper.cpp");
    }
    
    clFinish(command_queue);
    
    vector<double> results;
    
    for (int i = 0; i < int(events.size()); i++)
    {
      results.push_back(event_to_time(events[i]));
    }
    
    result_summary summary;

    summary.metadata = ident;
    
    summary.E = gsl_stats_mean(&results[0], 1, results.size());
    summary.D = gsl_stats_sd(&results[0], 1, results.size());
    summary.V = gsl_stats_variance(&results[0], 1, results.size());
    summary.S = gsl_stats_skew(&results[0], 1, results.size());
    summary.K = gsl_stats_kurtosis(&results[0], 1, results.size());
    summary.n = results.size();
    summary.JB = summary.n/6.0 * (sqr(summary.S) + sqr(summary.K-3)/4.0);
    
    logfile << "E: " << summary.E << "us D:" << summary.D << "us  JB:" << summary.JB << endl;
    
    if (summary.JB < summary_final.JB)
    {
      summary_final = summary;
    }
    
    if (summary.JB < 6000) ///< glitches mess with the statistics, so we retry a few times if JB is too big
    {
      break;
    }
  }
  
  logfile << "final E: " << summary_final.E << "us D:" << summary_final.D << "us  JB:" << summary_final.JB << endl;
    
  this->results.push_back(summary_final);
}


void ocl_test::geterr(cl_int err)
{
  if (err == CL_SUCCESS)
  {
    return;
  }
  
  #define test_error(error) if (err == error) {logfile << "OpenCL error: " << #error << endl; throw runtime_error(#error);}
  
  test_error(CL_SUCCESS);
  test_error(CL_DEVICE_NOT_FOUND);
  test_error(CL_DEVICE_NOT_AVAILABLE);                     
  test_error(CL_COMPILER_NOT_AVAILABLE);                   
  test_error(CL_MEM_OBJECT_ALLOCATION_FAILURE);            
  test_error(CL_OUT_OF_RESOURCES);                         
  test_error(CL_OUT_OF_HOST_MEMORY);                       
  test_error(CL_PROFILING_INFO_NOT_AVAILABLE);             
  test_error(CL_MEM_COPY_OVERLAP);                         
  test_error(CL_IMAGE_FORMAT_MISMATCH);                    
  test_error(CL_IMAGE_FORMAT_NOT_SUPPORTED);               
  test_error(CL_BUILD_PROGRAM_FAILURE);                    
  test_error(CL_MAP_FAILURE);                              
  test_error(CL_MISALIGNED_SUB_BUFFER_OFFSET);             
  test_error(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
  test_error(CL_INVALID_VALUE);                            
  test_error(CL_INVALID_DEVICE_TYPE);                      
  test_error(CL_INVALID_PLATFORM);                         
  test_error(CL_INVALID_DEVICE);                           
  test_error(CL_INVALID_CONTEXT);                          
  test_error(CL_INVALID_QUEUE_PROPERTIES);                 
  test_error(CL_INVALID_COMMAND_QUEUE);                    
  test_error(CL_INVALID_HOST_PTR);                         
  test_error(CL_INVALID_MEM_OBJECT);                       
  test_error(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);          
  test_error(CL_INVALID_IMAGE_SIZE);                       
  test_error(CL_INVALID_SAMPLER);                          
  test_error(CL_INVALID_BINARY);                           
  test_error(CL_INVALID_BUILD_OPTIONS);                    
  test_error(CL_INVALID_PROGRAM);                          
  test_error(CL_INVALID_PROGRAM_EXECUTABLE);               
  test_error(CL_INVALID_KERNEL_NAME);                      
  test_error(CL_INVALID_KERNEL_DEFINITION);                
  test_error(CL_INVALID_KERNEL);                           
  test_error(CL_INVALID_ARG_INDEX);                        
  test_error(CL_INVALID_ARG_VALUE);                        
  test_error(CL_INVALID_ARG_SIZE);                         
  test_error(CL_INVALID_KERNEL_ARGS);                      
  test_error(CL_INVALID_WORK_DIMENSION);                   
  test_error(CL_INVALID_WORK_GROUP_SIZE);                  
  test_error(CL_INVALID_WORK_ITEM_SIZE);                   
  test_error(CL_INVALID_GLOBAL_OFFSET);                    
  test_error(CL_INVALID_EVENT_WAIT_LIST);                  
  test_error(CL_INVALID_EVENT);                            
  test_error(CL_INVALID_OPERATION);                        
  test_error(CL_INVALID_GL_OBJECT);                        
  test_error(CL_INVALID_BUFFER_SIZE);                      
  test_error(CL_INVALID_MIP_LEVEL);                        
  test_error(CL_INVALID_GLOBAL_WORK_SIZE);                 
  test_error(CL_INVALID_PROPERTY);                         
}

void ocl_test::geterr(cl_int err, int line, const char* file)
{
  if (err == CL_SUCCESS)
  {
    return;
  }
  
  logfile << "Error at line " << line << " in file: " << file;
  geterr(err);
}

void ocl_test::alloc_memory()
{
  cl_int errnum = 0;
  dev_buffer1 = clCreateBuffer(context, CL_MEM_READ_WRITE, alloc_size, NULL, &errnum);
  geterr(errnum, __LINE__, "ocl_wrapper.cpp");
  dev_buffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE, alloc_size, NULL, &errnum);
  geterr(errnum, __LINE__, "ocl_wrapper.cpp");
  host_buffer_size = alloc_size / sizeof(int);
  dev_buffer_size = host_buffer_size;
  
  host_buffer1 = malloc(alloc_size);
  host_buffer2 = malloc(alloc_size);
  
  for (int i = 0; i < host_buffer_size; i++)
  {
    ((int*)host_buffer1)[i] = rand();
  }

  errnum = clEnqueueWriteBuffer(command_queue, dev_buffer1, CL_TRUE, 0, alloc_size, host_buffer1, 0, NULL, NULL);
  geterr(errnum, __LINE__, "ocl_wrapper.cpp");
  
}

void ocl_test::free_memory()
{
  free(host_buffer1);
  free(host_buffer2);
  
  clReleaseMemObject(dev_buffer1);
  clReleaseMemObject(dev_buffer2);
}

std::string ocl_test::clean_spaces(std::string name)
{
  for (int i = 0; i < int(name.length()); i++)
  {
    if (name[i] == ' ')
    {
      name[i] = '_';
    }
  }
  
  return name;
}

void ocl_test::export_to_text(std::string fname)
{
  ofstream f(fname.c_str());
  
  for (int i = 0; i < int(results.size()); i++)
  {
    result_summary sm = results[i];
    
    f << clean_spaces(sm.metadata.dev_name) << " " << clean_spaces(sm.metadata.test_name)
      << " " << sm.metadata.local_size[0] << " " << sm.metadata.local_size[1] << " " << sm.metadata.local_size[2]
      << " " << sm.metadata.global_size[0] << " " << sm.metadata.global_size[1] << " " << sm.metadata.global_size[2]
      << " " << sm.E
      << " " << sm.D
      << " " << sm.V
      << " " << sm.S
      << " " << sm.K
      << " " << sm.n
      << " " << sm.JB << endl;
  }
}

