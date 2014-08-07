package org.stream_gpu.knn.bucketselect;

import com.amd.aparapi.Kernel;

public class AssignBucket extends Kernel
{
	public float[] values;
	public int[] flags;
	public float min;
	public float range;
	public int bucket;
	public int num_buckets;
	
	public void run()
	{
		float min_b = bucket * range +min;
		float max_b = min_b + range;
		float val = values[getGlobalId()];
		boolean in_bucket = bucket < num_buckets-1 ? val >= min_b && val < max_b : val >= min_b; 
		
		if (in_bucket)
		{
			flags[getGlobalId()] = 1;
		}
		else
			flags[getGlobalId()] = 0;
	}

	public void update(float[] values, int found) {
		this.values  =values;
		this.flags = new int[ found ];
	}
}