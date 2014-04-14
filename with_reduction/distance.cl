//#pragma OPENCL EXTENSION cl_amd_printf : enable


__kernel void prepare_distance(__global const float* samples,
								__global const float* input,
							  __global float* result, 
								const int offset)
{        
	int id = get_global_id(0);
	float d = input[id] - samples[offset + id];
	result[id] = d*d;
} 

        
__kernel void reduction_scalar(__global float* data, 
      __local float* partial_sums) {

   int lid = get_local_id(0);
   int group_size = get_local_size(0);

   partial_sums[lid] = data[get_global_id(0)];
   barrier(CLK_LOCAL_MEM_FENCE);

   for(int i = group_size/2; i>0; i >>= 1) {
      if(lid < i) {
         partial_sums[lid] += partial_sums[lid + i];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(lid == 0) {
      data[get_group_id(0)] = partial_sums[0];
   }
}