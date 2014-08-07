package org.stream_gpu.knn.aparapi;

import com.amd.aparapi.Kernel;

class PrefixSumDownKernel extends Kernel 
{
	public long[] src;
	public int offset;
	public int stride;
	@Override
	public void run() {
	   	int id = offset + 2 *stride*getGlobalId(0);
		src[id+stride] += src[id];
	}
	
}