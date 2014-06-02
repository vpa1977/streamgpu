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

// run the prefix sum. 

// local_buf - half of the workgroup size
 __kernel void prefix_sum_group(__global uint* src, __local uint* local_buf)
 {
	uint stride, step;
	uint prev =0;
	uint id = get_global_id(0) << 1;
	uint lid = get_local_id(0);
	uint half_size = get_local_size(0) /2;
	

	local_buf[lid] = src[id] + src[id+1];
	barrier( CLK_LOCAL_MEM_FENCE);
	for (stride = 1, step =1 ; stride <=  half_size ; stride =stride <<1, ++step) 
	{
		// module power of 2, see testSequence in TestScanDigit.java
		int value = lid - prev;
		if (value >=0) 
		{
			uint pw = 1 << step;
			uint mod = value & (pw  -1);
			if (mod == 0 ) 
			{
				local_buf[lid+stride] += local_buf[lid];
			}
			barrier( CLK_LOCAL_MEM_FENCE);
		}
		prev += stride;
	}
	src[id+1] = local_buf[lid];
 }

// same code as prefix_sum_group, but uses last element of each workgroup, global size [0.. num_workgroups] for the prefix_sum_workgroup
// local_buf - half of the number of workgroups
 __kernel void prefix_sum_global(__global uint* src, __local uint* local_buf,const int global_stride)
 {
	uint stride, step;
	uint prev =0;
	uint id = 	2 * get_global_id(0) * global_stride + global_stride -1;
	uint lid = get_local_id(0);
	uint half_size = get_local_size(0) /2;
	local_buf[lid] = src[id] + src[id+global_stride];
	barrier( CLK_LOCAL_MEM_FENCE);
	for (stride = 1, step =1 ; stride <=  half_size ; stride =stride <<1, ++step) 
	{
		// module power of 2, see testSequence in TestScanDigit.java
		int value = lid - prev;
		if (value >=0) 
		{
			uint pw = 1 << step;
			uint mod = value & (pw  -1);
			if (mod == 0) 
			{
				local_buf[lid+stride] += local_buf[lid];
			}
			barrier( CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		}
		prev += stride;
	}
	src[id+global_stride] = local_buf[lid];
 }

 // second pass of the prefix sum - 
 __kernel void prefix_sum_group_down(__global uint* src, __local uint* local_buf) 
 {
	uint stride, step;
	uint id = get_global_id(0);
	uint lid = get_local_id(0);
	uint half_size = get_local_size(0) /2;
	uint prev =half_size -1;
	local_buf[lid] = src[id];
	barrier( CLK_LOCAL_MEM_FENCE);

	if ( lid <  get_local_size(0) -1)
	{
		for (stride = half_size/2, step = half_size; stride >0; stride = stride >> 1, step = step >> 1 )
		{
			int value = lid - prev;
			if (value >=0)
			{
				int mod = value & ( step -1);
				if (mod == 0)
				{
					local_buf[lid + stride] += local_buf[lid];
					barrier( CLK_LOCAL_MEM_FENCE);
				}
			}
			prev -= stride;
		}
	}
	src[id] = local_buf[lid];
 }

  __kernel void prefix_sum_global_down(__global uint* src, __local uint* local_buf,__global uint* indices, const int global_stride )
 {
	uint stride, step;
	
	int id = (2 * global_stride -1) + 2*global_stride  * get_global_id(0);
	int lid = 2*get_local_id(0);
	int half_size = get_local_size(0);
	
	local_buf[lid] = src[id];
	local_buf[lid+1] = src[id + global_stride];
	barrier( CLK_LOCAL_MEM_FENCE);
	for (stride = half_size/2; stride >= 1; stride = stride >> 1)
	{
		if (stride == indices[lid])
		{
			local_buf[lid] += local_buf[lid-stride];
		}
		if (stride == indices[lid+1]) 
		{
			local_buf[lid +1] += local_buf[lid+1-stride];
		}
		barrier( CLK_LOCAL_MEM_FENCE);
	}
	src[id] = local_buf[lid];
	src[id+global_stride] = local_buf[lid+1];
 }