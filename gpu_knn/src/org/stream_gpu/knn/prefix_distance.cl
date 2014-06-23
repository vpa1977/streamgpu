 
 //
 __kernel void distance( __global float* src_vector, // example vector 
 						 __global const float2* ranges, // ranges
 						 __global float* component_dst,  // 
 						 __global float* window, 
 						 const uint element, 
 						 const uint element_size,
 						 const uint numerics_size)
 {
 	uint offset = element*element_size;
 	uint id = get_global_id(0);
 	if (id < numerics_size) 
 	{
 		float2 range = ranges[get_local_id(0)];
		width = ( range.y - range.x);
		component_dst[id] = (src_vector[id] - range.x) / width  - (window[offsete + i] - range.x)/width;
 	}
 	else
 	{
 		component_dst[id] = isnotequal( input[i] , samples[vector_offset + i]);
 	}
 }
 
 
 __kernel void prefix_sum_up(__global uint* src, __local uint* local_buf, __global uint* up_indices,  const uint max_up) 
 {
   	int id = 2*get_global_id(0)+1;
   	int lid = get_local_id(0);
 	int up_index =  up_indices[id];
 	int half_size = get_local_size(0)/2;
 	int step =2;
 	int stride;
 	local_buf[lid] = src[id] + src[id-1];
 	barrier( CLK_LOCAL_MEM_FENCE);
 	// perform up sweep
  	for (stride = 1; stride <= half_size; stride = stride *2, ++step)
  	{
  		if (up_index >= step)
  		{
  			local_buf[lid] += local_buf[lid - stride];
  		}
  		barrier( CLK_LOCAL_MEM_FENCE);
  	}
  	
  	stride = stride * 2;
  	src[id] = local_buf[lid];
  	barrier( CLK_GLOBAL_MEM_FENCE);
  	
  	for (; step <= max_up; ++step, stride = stride * 2 )
  	{
  		if (up_index >= step)
  		{
  			src[id] += src[id - stride];
  		}
  		barrier( CLK_GLOBAL_MEM_FENCE );
  	}
 }