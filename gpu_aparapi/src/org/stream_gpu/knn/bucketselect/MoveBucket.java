package org.stream_gpu.knn.bucketselect;

import com.amd.aparapi.Kernel;

public class MoveBucket extends Kernel {
	public float[] values;
	public float[] output;
	public int[] flags;
	public float min;
	public float range;
	public int bucket;
	public int num_buckets;
	
	
	
	
	public void initialize(float[] values, float min, float range, int bucket, int num_buckets, int[] flags, int found)
	{
		this.values = new float[values.length];
		System.arraycopy(values, 0, this.values, 0, this.values.length);
		this.output = new float[found];
		this.min = min;
		this.range = range; 
		this.bucket = bucket;
		this.num_buckets = num_buckets;
		this.flags = flags;
		//this.setExecutionMode(EXECUTION_MODE.JTP);
	}
	
	
	public void run()
	{
		int id = getGlobalId();
		
		float min_b = bucket * range +min;
		float max_b = (bucket+1)*range + min;
		float val = values[id];
		int offset = flags[id];
		if (val >= min_b && val < max_b)
		{
			output[offset] = values[id];
		}
	}


}
