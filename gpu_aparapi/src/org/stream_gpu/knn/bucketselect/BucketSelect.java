package org.stream_gpu.knn.bucketselect;

import java.util.Random;

import weka.classifiers.trees.m5.Values;

import com.amd.aparapi.Device;
import com.amd.aparapi.Range;
import com.amd.aparapi.Kernel.EXECUTION_MODE;

public class BucketSelect {
	
	private PrefixSum sum;
	private AssignBucket assign;
	private MoveBucket move;
	private ReadResult readResult;
	public BucketSelect(Device dev)
	{
		sum = new PrefixSum(dev);
		assign = new AssignBucket();
		move = new MoveBucket();
		readResult = new ReadResult();
		assign.num_buckets = 2;

	}
	
	public float[] select(Device dev, float[] in, int k)
	{
		float min = 0;
		float max = Float.MAX_VALUE;
		
		
		assign.flags = new int[in.length];
		assign.values = in;
		assign.min = min;
		assign.range =  (max-min)/assign.num_buckets;
		
		readResult.dst = new float[k];
		
		move.min = min;
		move.range = assign.range;
		
		
		Range r = dev.createRange(in.length);
		
		boolean done = false;
		int bucket = 0;
		int found = 0;
		int dst_offset = 0;
		while (!done)
		{
			
			assign.bucket = bucket;
			//assign.setExecutionMode(EXECUTION_MODE.JTP);
			assign.execute(r);
			int[] prefix = sum.prefixSum(dev, Math.min(dev.getMaxWorkGroupSize(), assign.flags.length), assign.flags);
			found = prefix[ r.getGlobalSize(0) -1];
			if (found > k)
			{
				if (found < in.length-1)
				{
					
					move.initialize(assign.values, min, assign.range, bucket, assign.num_buckets, prefix, found+1);
					move.execute(r);
					assign.update(move.output, found+1);
					r = dev.createRange(found+1);
				}
				double range = (max - min)/ assign.num_buckets;
				min = min + (float)(range * bucket);
				max = min + (float)range;
				assign.range =(float) range;

			}
			else
			{
				// check found == 1 condition.
				
				if (found > 0)
				{
					k-=(found+1);
					readResult.initialize(assign.values, prefix, min,assign.range, bucket, assign.num_buckets, dst_offset);
					dst_offset += (found+1);
					readResult.execute(r);
				}
				++bucket;
				if (bucket > assign.num_buckets)
					throw new RuntimeException("too many buckets");
				
				if (k <= 0)
					done = true;
			}
		}
		
		//System.out.println(step + " found "+ found);
		
		return readResult.dst;
	}
	
	public static void main(String[] args)
	{
		//System.setProperty("com.amd.aparapi.enableShowGeneratedOpenCL", "true");
		Device gpu = Device.firstGPU();
		float[] test_array = new float[1024];
		Random rnd = new Random();
		for (int i = 0 ;i < test_array.length; ++i)
			test_array[i] = test_array.length - i;


		BucketSelect select = new BucketSelect(gpu);
		
		float[] result = select.select(gpu, test_array , 32 );
		System.out.println();
/*		select.select(gpu, test_array , 32 );
		
		long start = System.currentTimeMillis();
		//for (int i = 0 ; i < 1000 ; i ++)
			select.select(gpu, test_array , 32 );
		long end = System.currentTimeMillis();
		System.out.println((end -start)/1000);
*/		
	}
}
