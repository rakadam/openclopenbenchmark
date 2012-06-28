#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <ocl_wrapper.h>

using namespace std;

int main(int argc, char* argv[])
{
    ocl_test test;
    
    test.run_tests_on_all();
    
    test.export_to_text("meas.txt");
}
