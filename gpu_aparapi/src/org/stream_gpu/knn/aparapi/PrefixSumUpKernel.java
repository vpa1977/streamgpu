package org.stream_gpu.knn.aparapi;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.Local;

class PrefixSumUpKernel extends Kernel 
{

	public long[] src;
	@Local
	public long[] local_buf;
	public int[] up_indices;
	public int max_up;
	
	@Override
	public void run() {
	   	int id = 2*getGlobalId(0)+1;
	   	int lid = getLocalId(0);
	 	int up_index =  up_indices[id];
	 	int half_size = getLocalSize(0)/2;
	 	int step =2;
	 	int stride;
	 	local_buf[lid] = src[id] + src[id-1];
	 	localBarrier();
	 	// perform up sweep
	  	for (stride = 1; stride <= half_size; stride = stride *2, ++step)
	  	{
	  		if (up_index >= step)
	  			local_buf[lid] += local_buf[lid - stride];
	  		localBarrier();
	  	}
	  	
	  	stride = stride * 2;
	  	src[id] = local_buf[lid];
	  	
	  	for (; step <= max_up; ++step, stride = stride * 2 )
	  	{
	  		if (up_index >= step)
	  			src[id] += src[id - stride];
	  	}
	}
	
}