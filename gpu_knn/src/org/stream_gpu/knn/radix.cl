#pragma OPENCL EXTENSION cl_amd_printf : enable

#define NUM_DIGITS 16

// next optimizaiton - use floats
__kernel void scan_digit(__global uint* src,  const int shift, 
						 __global uint* global_counts, __local uint* local_counts) 
{
	uint id = get_local_id(0);
	uint grp_id = get_group_id(0);
	uint grp_size = get_local_size(0);
	
	uint mask = 0xF << shift;
	uint value = (src[ get_local_id(0) ] & mask) << shift;
	local_counts[ value ] +=1;
	barrier (CLK_LOCAL_MEM_FENCE)
	if (id == 0) {
		uint offset = NUM_DIGITS * grp_id;
		global_counts[ offset ] = local_counts[0];
		for (int i = 1;i < 16; ++i )  {
			local_counts[i] += local_counts[i-1];
			global_counts[ offset +  i ] = local_counts[i];
		}
	}
}


__kernel void prefix_sum0(__global uint* src, const int stage) 
{
	uint digit;
	for (digit = 0; digit < NUM_DIGITS; ++digit) 
	{
		
	}
}

__kernel void prefix_sum(__global uint* src, const int stage) 
{
	uint global_start;
	uint gloabl_offset;
	global_start = (get_group_id(0) + (get_group_id(0)/stage)*stage) *
                   get_local_size(0) + get_local_id(0);
	global_offset = stage * get_local_size(0);

	src[global_start + global_offset] += src[global_start];
}