#pragma OPENCL EXTENSION cl_amd_printf : enable

__kernel void local_distance(__global const float* input,
						__global const float* samples,
						__global const float2* ranges,
						__global float* result,
						__local float* input_adj, 
						__local float* input_w,
						__local float* temp,
						const int element_count,
						const int numerics_size)
{
	int global_id = get_global_id(0);
	int id = get_local_id(0) % element_count;
	int local_id = get_local_id(0);
	input_w[id] = (ranges[id].y - ranges[id].x);
	input_adj[id] = id < numerics_size ? (input[id] - ranges[id].x)/input_w[id] : input[id];
	barrier( CLK_LOCAL_MEM_FENCE );
	temp[local_id] =  id < numerics_size ?  input_adj[id] - (samples[global_id] - ranges[id].x)/input_w[id] : isnotequal( input[id] , samples[global_id]);
	temp[local_id] = temp[local_id] * temp[local_id];
	barrier( CLK_LOCAL_MEM_FENCE );
	//if (id == 0 ) 
	//{
	//	for (id=get_local_id(0)+1; id< element_count; id++)
	//		temp[local_id] += temp[id];
	//	result[global_id / element_count] = temp[local_id];
	//}
}
