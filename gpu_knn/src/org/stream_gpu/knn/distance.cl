#pragma OPENCL EXTENSION cl_amd_printf : enable

/*
   distance kernel one
*/
__kernel void square_distance(__global const float* input,
						__global const float* samples,
						__global const float2* ranges,
						__global float* result, 
						const int element_count, 
						const int numerics_size, 
						const int nominal_size)
{        
	int result_offset = get_global_id(0);
	int vector_offset = element_count * result_offset;
	float point_distance = 0;
	float val;
	float width;
	
	
	
	for (int i = 0; i < numerics_size ; i ++ ) 
	{
		float2 range = ranges[i];
		width = ( range.y - range.x);
		val = (input[i] - range.x) / width  - (samples[ vector_offset + i] - range.x)/width;
		point_distance += val*val; 
	}
	
	for (int i = numerics_size; i < element_count; i ++ ) 
	{
		point_distance += isnotequal( input[i] , samples[vector_offset + i]);
	}
	result[result_offset] = point_distance;
}         



