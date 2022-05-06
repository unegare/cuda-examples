#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "../CGBN/samples/utility/gpu_support.h"
#include "../CGBN/samples/utility/cpu_support.h"

#define TPI 32
#define BITS 1024

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

typedef struct {
    context_t bn_context;
    env_t bn_env;
    void *data;
    cgbn_error_report_t *report;
} total_context_t;

template<uint32_t from_bits, uint32_t to_bits>
__host__ __device__ cgbn_mem_t<to_bits> convert(const cgbn_mem_t<from_bits>& from) {
  if constexpr (to_bits < from_bits) {
    cgbn_mem_t<to_bits> res;
    memcpy(res._limbs, from._limbs, std::size(res._limbs) * sizeof(res._limbs[0]));
    return res;
  } else if constexpr (to_bits == from_bits) {
    return from;
  } else {
    cgbn_mem_t<to_bits> res;
    memcpy(res._limbs, from._limbs, std::size(from._limbs) * sizeof(from._limbs[0]));
    for (size_t i = std::size(from._limbs); i < std::size(res._limbs); i++) {
      res._limbs[i] = 0;
    }
    return res;
  }
}

//typedef cgbn_mem_t<1024> amount_t;
//typedef cgbn_mem_t<1024> cgbn_1024_t;

using amount_t = cgbn_mem_t<1024>;
using cgbn_1024_t = cgbn_mem_t<1024>;

typedef struct {
  amount_t source_amount;
  amount_t target_amount;
  uint32_t source;
  uint32_t target;
  uint32_t agent;
} SampleExchange;

typedef struct {
  amount_t source_amount;
  amount_t target_amount;
  uint32_t source;
  uint32_t target;
  uint32_t agent;
  uint32_t fee_numer;
  uint32_t fee_denom;
  
  __host__ __device__ void source_to_target(total_context_t& ctx, amount_t &_s, amount_t &_t) {
    env_t::cgbn_t sa, ta, s, t, numer, denom;

    cgbn_load(ctx.bn_env, s, &_s);
    cgbn_load(ctx.bn_env, sa, &source_amount); 
    cgbn_load(ctx.bn_env, ta, &target_amount); 

    cgbn_mul(ctx.bn_env, numer, ta, s);
    cgbn_add(ctx.bn_env, denom, sa, s);
    cgbn_div(ctx.bn_env, t, numer, denom);

    cgbn_store(ctx.bn_env, &_t, t);
  }
  __host__ __device__ void target_to_source(total_context_t& ctx, amount_t &_t, amount_t &_s) {
    env_t::cgbn_t sa, ta, s, t, numer, denom;

    cgbn_load(ctx.bn_env, t, &_t);
    cgbn_load(ctx.bn_env, sa, &source_amount);
    cgbn_load(ctx.bn_env, ta, &target_amount);

    if (cgbn_compare(ctx.bn_env, ta, t) <= 0) {
      printf("less than zero\n");
      printf("cgbn_compare: %d\n", cgbn_compare(ctx.bn_env, ta, t));
      cgbn_set_ui32(ctx.bn_env, s, 0);
      cgbn_store(ctx.bn_env, &_s, s);
      return;
    }

    cgbn_mul(ctx.bn_env, numer, sa, t);
    cgbn_sub(ctx.bn_env, denom, ta, t);
    cgbn_div(ctx.bn_env, s, numer, denom);

    cgbn_store(ctx.bn_env, &_s, s);
  }
} BaseAMMDescriptor;

typedef struct BaseAMMDescriptor_MPZ {
  mpz_t source_amount;
  mpz_t target_amount;
  uint32_t source;
  uint32_t target;
  uint32_t agent;

  BaseAMMDescriptor_MPZ() {
    mpz_init(source_amount);
    mpz_init(target_amount);
  }
  BaseAMMDescriptor_MPZ(const char s[], const char t[]) {
    mpz_init_set_str(source_amount, s, 10);
    mpz_init_set_str(target_amount, t, 10);
  }
  ~BaseAMMDescriptor_MPZ() {
    mpz_clear(source_amount);
    mpz_clear(target_amount);
  }
  void to_device_type(BaseAMMDescriptor *bammd) {
    from_mpz(source_amount, bammd->source_amount._limbs, std::size(bammd->source_amount._limbs));
    from_mpz(target_amount, bammd->target_amount._limbs, std::size(bammd->target_amount._limbs));
    bammd->source = source;
    bammd->target = target;
    bammd->agent = agent;
  }
} BaseAMMDescriptor_MPZ;

__global__ void cuda_kernel(cgbn_error_report_t *report, BaseAMMDescriptor *bammd, amount_t *req_source, amount_t *req_target, uint32_t count) {
  context_t bn_context(cgbn_report_monitor, report, 0);
  env_t bn_env(bn_context.env<env_t>());

  total_context_t ctx{bn_context, bn_env, bammd, report};

//  bammd->source_to_target(ctx, *req_source, *req_target);
  bammd->target_to_source(ctx, *req_source, *req_target);
}

void cuda_wrapper() {
  cgbn_error_report_t *report;

  BaseAMMDescriptor bammd;
  BaseAMMDescriptor *gpuBammd;

  amount_t reqSource, reqTarget;
  amount_t *gpuReqSource, *gpuReqTarget;

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  CUDA_CHECK(cudaMalloc((void**)&gpuBammd, sizeof(BaseAMMDescriptor)));
  CUDA_CHECK(cudaMalloc((void**)&gpuReqSource, sizeof(amount_t)));
  CUDA_CHECK(cudaMalloc((void**)&gpuReqTarget, sizeof(amount_t)));


  BaseAMMDescriptor_MPZ bammd_mpz("6000000", "3000000");

//  mpz_t sa, ta;
  mpz_t s, t;
//  mpz_init(sa);
//  mpz_init(ta);
  mpz_init(s);
  mpz_init(t);

//  mpz_init_set_str(sa, "6000000", 10);
//  mpz_init_set_str(ta, "3000000", 10);
  mpz_init_set_str(s, "1500000", 10);

//  from_mpz(sa, bammd.source_amount._limbs, std::size(bammd.source_amount._limbs));
//  from_mpz(ta, bammd.target_amount._limbs, std::size(bammd.source_amount._limbs));
  bammd_mpz.to_device_type(&bammd);

  from_mpz(s, reqSource._limbs, std::size(bammd.source_amount._limbs));

  CUDA_CHECK(cudaMemcpy(gpuBammd, &bammd, sizeof(BaseAMMDescriptor), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(gpuReqSource, &reqSource, sizeof(amount_t), cudaMemcpyHostToDevice));

  cuda_kernel<<<1,TPI>>>(report, gpuBammd, gpuReqSource, gpuReqTarget, 0);

  CUDA_CHECK(cudaMemcpy(&reqTarget, gpuReqTarget, sizeof(amount_t), cudaMemcpyDeviceToHost));

  to_mpz(t, reqTarget._limbs, std::size(reqTarget._limbs));

  char *t_str = mpz_get_str(NULL, 10, t);
  char *s_str = mpz_get_str(NULL, 10, s);

  std::cout << s_str << " -> " << t_str << std::endl;

  void (*freefunc)(void *, size_t);
  mp_get_memory_functions (NULL, NULL, &freefunc);

  freefunc(t_str, strlen(t_str) +1);
  freefunc(s_str, strlen(s_str) +1);

//  mpz_clear(sa);
//  mpz_clear(ta);
  mpz_clear(s);
  mpz_clear(t);
  CUDA_CHECK(cudaFree(gpuBammd));
  CUDA_CHECK(cudaFree(gpuReqSource));
  CUDA_CHECK(cudaFree(gpuReqTarget));
}

int main() {
  cuda_wrapper();
  return 0;
}
