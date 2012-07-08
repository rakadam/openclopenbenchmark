#ifndef OCL_WRAPPER_H
#define OCL_WRAPPER_H
#include <CL/cl.h>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <set>

class ocl_test;

typedef void (*test_func)(ocl_test&);

extern cl_ulong local_mem_size;

enum kernel_flags{
  KERNEL_FLAG_ALL_LOCAL_SIZES = 1,
  KERNEL_FLAG_MORE_LOCAL_SIZES = 2
};

class ocl_test
{
  struct test_iden
  {
    std::string dev_name;
    std::string test_name;
    size_t local_size[3];
    size_t global_size[3];
  };
  
  std::set<int> bad_kernels;
  std::vector<test_func> gold_test_funcs;
  std::vector<test_func> ocl_test_funcs;
  std::map<std::string, test_func> gold_test_funcs_by_name;
  std::map<std::string, test_func> ocl_test_funcs_by_name;
  std::map<test_func, std::string> ocl_test_name_by_func;
  
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
  
  bool dummy_run;
  
  std::string clean_spaces(std::string name);
  bool interesting_number(long num, std::string name);
public:
  
  std::map<std::string, std::map<int, int> > max_global_size; ///< max global size for the kernels, if ==zero, then it is unlimited, if -1 it's the max simultanious hw thread num
  std::map<std::string, std::map<int, int> > max_local_size; ///< max local size for the kernels, if ==zero, then it is unlimited
  std::map<std::string, std::map<int, int> > min_local_size; ///< min local size for the kernels, (default)zero is ignored
  std::map<std::string, unsigned> kernel_flags; ///< or-ed flags of kernel_flags
  
  cl_mem dev_buffer1, dev_buffer2;
  cl_int dev_buffer_size; ///< assumes 32bit elemsize

  void *host_buffer1, *host_buffer2;
  int host_buffer_size; ///< assumes 32bit elemsize

  ocl_test();
  void compile_test();
  void get_max_sizes();
  void run_tests();
  void run_tests_on_all();
  double event_to_time(cl_event event);
  void test_configuration(cl_kernel kernel, test_iden ident);
  void register_ocl_test(test_func, std::string name);
  void register_gold_test(test_func, std::string name);
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
