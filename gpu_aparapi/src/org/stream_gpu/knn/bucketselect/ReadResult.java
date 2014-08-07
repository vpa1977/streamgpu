package org.stream_gpu.knn.bucketselect;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.EXECUTION_MODE;

public class ReadResult extends Kernel {
	public float[] src;
	public float[] dst;
	public int[] flags;
	
	public float min;
	public float range;
	public int bucket;
	public int num_buckets;
	public int start;
	
	public void initialize(float[] src, int[] flags, float min, float range, int bucket, int num_buckets, int start)
	{
		this.src = src;
		this.flags = flags;
		this.min = min;
		this.range = range;
		this.bucket = bucket; 
		this.num_buckets = num_buckets;
		this.start = start;
		//this.setExecutionMode(EXECUTION_MODE.JTP);
	}

	
	public void run() 
	{
		int id = getGlobalId();
		
		float min_b = bucket * range +min;
		float max_b = (bucket+1)*range + min;
		float val = src[id];
		int offset = flags[ id];
		if (val >= min_b && val < max_b)
		{
			dst[offset + start]  = val;
			src[id] = Float.MAX_VALUE;
		}
		
	}
}
