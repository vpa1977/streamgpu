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
	}
	barrier(CLK_LOCAL_MEM_FENCE);

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
 
 __kernel void prefix_sum_down(__global uint* src,   const uint max_down) 
 {
 	int stride = 1 << (max_down);
   	int id = stride*get_global_id(0) -1;
	src[id+(stride>>1)] += src[id];
}



#define k 32
//								  // __global uint* dst,
 
 __kernel void scan_and_move_digit(__global uint* src,  // source ints
 							__global uint* dst,
							 const int shift,  // bit shift
						 __global uint* global_counts, // counts of digits 0=> [0 ... workgroup_count] 1=>[ workgroup_count +1... workgroup_count + workgroup_count] etc
						  __local uint* local_counts)  // local counts 0=>[0.. workgroup_size] 1 => [workgroup_size +1 ; workgroup_size + workgroup_size] etc
{
	uint workgroup = get_group_id(0);
	uint num_workgroups = get_num_groups(0);
	uint group_size = get_local_size(0)*2;
	uint id = 2*get_global_id(0);
	uint local_index = 2*get_local_id(0);
	uint mask =  (0xF << shift);
	uint digit, digit1;
	uint i; 
	uint offset;
	// reset all local memory to 0
	for (digit=0; digit < NUM_DIGITS; digit ++ ) 
	{
		offset = digit * group_size;
		barrier(CLK_LOCAL_MEM_FENCE);
		local_counts[offset+local_index]=0;
		local_counts[offset+local_index+1]=0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	

	digit =  (src[id] & mask) >> shift;
	
	// scan algorithm
	// scan order - digit at digit_pos for every element of the vector
	// local_counts - bit vectors local_count[0 ... workgroup_size] [ 0 .. NUM_DIGITS]
	offset = digit * group_size + local_index;
	local_counts[offset]=1;

	digit1 =  (src[id+1] & mask) >> shift;
	offset = digit1 * group_size + local_index+1;
	local_counts[offset]=1;

	
	for (uint cur_digit=0; cur_digit < NUM_DIGITS; cur_digit ++ ) 
	{
		uint global_offset = cur_digit * group_size;
        int tid = get_local_id(0);
//  see https://code.google.com/p/clpp/source/browse/trunk/src/clpp/clppScan_Default.cl?r=126        
	    offset = 1;
		// Build the sum in place up the tree
		for(int d = group_size>>1; d > 0; d >>=1)
		 {
		    barrier(CLK_LOCAL_MEM_FENCE);
		    if(tid<d)
		    {
		            int ai = offset*(2*tid + 1) - 1;
		            int bi = offset*(2*tid + 2) - 1;
		            local_counts[bi+global_offset] += local_counts[ai+global_offset];
		    }
		    offset *= 2;
        }
		
	    // scan back down the tree
	    // Clear the last element
        if(tid == 0)
             local_counts[group_size - 1] = 0;
		
	    // traverse down the tree building the scan in the place
	    for(int d = 1; d < group_size ; d *= 2)
	    {
		    offset >>=1;
		    barrier(CLK_LOCAL_MEM_FENCE);
		                
		    if(tid < d)
		    {
		            int ai = offset*(2*tid + 1) - 1;
		            int bi = offset*(2*tid + 2) - 1;
		                        
		            float t = local_counts[ai+global_offset];
		            local_counts[ai+global_offset] = local_counts[bi+global_offset];
		            local_counts[bi+global_offset] += t;
		    }
	     }
        
	}
	offset = local_counts[ digit * group_size + local_index] + global_counts[ digit * num_workgroups + get_group_id(0)];
	if (offset <= k)
		dst[ offset ] = src[id];
	else
		dst[ offset ] = 0;

	offset = local_counts[ digit1 * group_size + local_index] + global_counts[ digit * num_workgroups + get_group_id(0)];
	
	if (offset <= k)
		dst[ offset ] = src[id];
	else
		dst[ offset ] = 0;
	
}
 

