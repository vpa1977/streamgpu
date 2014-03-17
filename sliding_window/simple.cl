/*
  Really dumb kernel, which computes average of several vectors 
*/

__kernel void addFloats(__global const float* input, 
						const int window_size, 
						const int element_count, 
						__global float* output)
{        
	int i;                                                                                            
	int offset = get_global_id(0);
	int element_offset = get_global_id(1);
	if (element_offset == 0)
		output[offset] = 0;
	output[offset] += input[ offset + element_offset*element_count]/window_size;
                                                        
}          