#ifndef OCL_WRAPPER_H
#define OCL_WRAPPER_H
#include <CL/cl.h>
#include <map>
#include <vector>
#include <string>
#include <fstream>

class ocl_test;

typedef void (*test_func)(ocl_test&);

extern cl_ulong local_mem_size;

class ocl_test
{
  struct test_iden
  {
    std::string dev_name;
    std::string test_name;
    size_t local_size[3];
    size_t global_size[3];
  };
  
  std::vector<test_func> gold_test_funcs;
  std::vector<test_func> ocl_test_funcs;
  
  typedef std::string Tdevicename;
  
  struct result_summary
  {
    test_iden metadata;
    double E; ///< mean
    double D; ///< standard deviance
    double V; ///< variance
    double S; ///< skewness
    double K; ///< kurtosis
    double n; ///< number of samples
    double JB; ///< result of the Jarque–Bera test (of how much gaussian the data is)
  };
  
  std::vector<result_summary> results;
  
  std::string dev_name;
  cl_command_queue command_queue;
  cl_context context;
  std::vector<cl_device_id> devices;
  int cur_dev_num;
  std::ofstream logfile;
  int alloc_size;

  cl_uint max_compute_units;
  cl_uint max_work_item_dimensions;
  size_t max_work_item_sizes[1024];
  size_t max_work_group_size;
  
  std::string clean_spaces(std::string name);
public:
  
  std::map<std::string, std::map<int, int> > max_global_size; ///< max global size for kernel, if ==zero, then it is unlimited, if -1 it's the max simultanious hw thread num
  std::map<std::string, std::map<int, int> > max_local_size; ///< max local size for kernel, if ==zero, then it is unlimited
  
  cl_mem dev_buffer1, dev_buffer2;
  cl_int dev_buffer_size; ///< assumes 32bit elemsize

  void *host_buffer1, *host_buffer2;
  int host_buffer_size; ///< assumes 32bit elemsize

  ocl_test();
  void get_max_sizes();
  void run_tests();
  void run_tests_on_all();
  double event_to_time(cl_event event);
  void test_configuration(cl_kernel kernel, test_iden ident);
  void register_ocl_test(test_func);
  void register_gold_test(test_func);
  cl_program ocl_load_src(const char* src);
  void launch_kernel(cl_kernel kernel, const char* name);
  void geterr(cl_int err);
  void geterr(cl_int err, int line, const char* file);
  void register_tests();
  void alloc_memory();
  void free_memory();
  void export_to_text(std::string fname);
};


#endif