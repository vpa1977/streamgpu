#pragma OPENCL EXTENSION cl_amd_printf : enable

#define NUM_DIGITS 16

 __kernel void scan_digit(__global uint* src,  // source ints
							 const int shift,  // bit shift
						 __global uint* global_counts, // counts of digits 0=> [0 ... workgroup_count] 1=>[ workgroup_count +1... workgroup_count + workgroup_count] etc
						  __local ushort* local_counts)  // local counts 0=>[0.. workgroup_size] 1 => [workgroup_size +1 ; workgroup_size + workgroup_size] etc
{
	uint workgroup = get_group_id(0);
	uint num_workgroups = get_num_groups(0);
	uint workgroup_size = get_local_size(0);
	uint local_index = get_local_id(0);
	uint mask =  (0xF << shift);
	uint digit;
	uint i; 
	uint offset;
	// reset all local memory to 0
	for (digit=0; digit < NUM_DIGITS; digit ++ ) 
	{
		offset = digit * workgroup_size;
		local_counts[offset+local_index]=0;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	digit =  (src[get_global_id(0)] & mask) >> shift;
	// scan algorithm
	// scan order - digit at digit_pos for every element of the vector
	// local_counts - bit vectors local_count[0 ... workgroup_size] [ 0 .. NUM_DIGITS]
	offset = digit * workgroup_size + local_index;
	local_counts[offset]=1;
	
	 barrier(CLK_LOCAL_MEM_FENCE);
	// local_counts reduced into global_counts
	// global_counts  offset vector for each digit/ workgroup [0 ... num_workgroups] [ 0.. NUM_DIGITS]
 	 for (digit = 0 ;digit < NUM_DIGITS ; digit ++)
	 {
		offset = digit * workgroup_size;
		for(i = workgroup_size/2; i>0; i >>= 1) {
			 if(local_index < i) {
				local_counts[offset+ local_index] += local_counts[offset + local_index + i];
			 }
			barrier(CLK_LOCAL_MEM_FENCE);
		}

	   if (local_index ==0)
	   {
			global_counts[ num_workgroups * digit + workgroup] = local_counts[offset];
	   }
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
 
 __kernel void prefix_sum_down(__global uint* src, __local uint* local_buf, __global uint* down_indices,   const uint max_down) 
 {
   	int id = get_global_id(0)+1;
   	int lid = get_local_id(0);
 	int down_index = down_indices[id];
 	int half_size = get_local_size(0)/2;
 	int quarter_size = half_size/2;
 	int step =1;
 	int stride;
  	// perform down-sweep
  	stride = ( 1 << (max_down-1));
  	for (step = max_down; stride >quarter_size ; --step, stride = stride /2)
  	{
  		if (down_index == step) 
  		{
  			src[id] += src[id - stride];
  			printf("global step %d assign %d from %d\n", step, id, id-stride);
  		}
  		barrier( CLK_GLOBAL_MEM_FENCE);
  	}  

 	local_buf[lid] = src[id];
 	printf("do copy %d value %d\n", id, local_buf[lid]);
 	barrier( CLK_LOCAL_MEM_FENCE );
  	
  	for (; stride >0; --step, stride = stride /2)
  	{
  		if (down_index == step) 
  		{
  			local_buf[lid] = local_buf[lid - stride] + local_buf[lid];
  			printf("local step %d assign %d from %d value %d+%d\n", step, lid, lid-stride, local_buf[lid - stride] ,local_buf[lid]);
  		}
  		barrier( CLK_LOCAL_MEM_FENCE);
  	}
	src[id] = local_buf[lid];
}
 
 