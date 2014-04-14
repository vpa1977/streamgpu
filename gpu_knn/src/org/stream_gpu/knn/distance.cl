#pragma OPENCL EXTENSION cl_amd_printf : enable

/*
   distance kernel one
*/
__kernel void square_distance(__global const float* input,
						__global const float* samples,
						__global float* result, 
						const int element_count)
{        
	int result_offset = get_global_id(0);
	int vector_offset = element_count * result_offset;
	float point_distance = 0;
	float val;
	for (int i = 0; i < element_count ; i ++ ) 
	{
		val = input[i] - samples[ vector_offset + i];
		point_distance += val*val; 
	}
	result[result_offset] = point_distance;
}         



