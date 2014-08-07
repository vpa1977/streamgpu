package org.stream_gpu.knn.aparapi;

import com.amd.aparapi.Kernel;

public abstract class Util extends Kernel{
	
	@Local
	public int[] local_counts;
	/** 
	 * 
	 * @param workgroup_size
	 * @param local_index - 2 * getLocalId()
	 * @param offset - N * workgroup_size
	 */
	public void reduce(int workgroup_size, int local_index, int offset) {
		int i;
		local_counts[offset + local_index] += local_counts[offset + local_index+1];
	 	localBarrier();
		
		for(i = workgroup_size/2; i>1; i >>= 1) {
		    localBarrier();
			 if (local_index < i ) {
			 	int a = local_counts[offset+ local_index];
			 	int b = local_counts[offset + local_index + i];
			 	local_counts[offset+ local_index] =a + b ;
			 }
		}
	}

	@Override
	public void run() {
		// TODO Auto-generated method stub
		
	}

}
