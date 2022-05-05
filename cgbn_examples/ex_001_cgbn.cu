#include <iostream>

#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../CGBN/samples/utility/gpu_support.h"

#define TPI 32
#define BITS 1024
#define INSTANCES 100'000

typedef struct {
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> sum;
} instance_t;

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

__global__ void cuda_kernel(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
  int32_t instance = 0;
  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t bn_env(bn_context.env<env_t>());
  env_t::cgbn_t a, b, r;

  cgbn_load(bn_env, a, &(instances[instance].a));
  cgbn_load(bn_env, b, &(instances[instance].b));
  cgbn_add(bn_env, r, a, b);
  cgbn_store(bn_env, &(instances[instance].sum), r);
}

void cuda_wrapper() {
  instance_t *instances, *gpuInstances;
  cgbn_error_report_t *report;

  instances = (instance_t*)malloc(sizeof(instance_t)*INSTANCES);
  if (!instances) {
    printf("mallocError\n");
    return;
  }

  for (size_t i = 0; i < std::size(instances[0].a._limbs); i++) {
    instances[0].a._limbs[i] = i;
    instances[0].b._limbs[i] = i;
  }

//  instances[0].a._limbs[0] = 0x80'00'00'00;
//  instances[0].b._limbs[0] = 0x80'00'00'00;

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void**)&gpuInstances, sizeof(instance_t)*INSTANCES));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*INSTANCES, cudaMemcpyHostToDevice));

  CUDA_CHECK(cgbn_error_report_alloc(&report));

  cuda_kernel<<<1,TPI>>>(report, gpuInstances, INSTANCES);

  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*INSTANCES, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < std::size(instances[0].sum._limbs); i++) {
    std::cout << instances[0].sum._limbs[i] << '\n';
  }
  std::cout << std::flush;

  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));
}

int main() {
  cuda_wrapper();
  return 0;
}
