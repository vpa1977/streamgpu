package org.stream_gpu.knn;

import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLQueue;

public class PrefixScan {
	
	private CLQueue m_queue;
	private CLKernel m_down_kernel;
	private CLKernel m_up_kernel;

	public PrefixScan(CLQueue queue, CLKernel down_sweep, CLKernel up_sweep)
	{
		m_queue = queue;
		m_down_kernel = down_sweep; 
		m_up_kernel = up_sweep;
	}
	
	public void scan( int global_size , int local_size )
	{
		int half_size = global_size/2;
		
		int n_stages = global_size / local_size;
		
		for (int stage =1 ; stage < n_stages ; stage <<=1)
		{
			m_down_kernel.setArg(0, stage);
			m_down_kernel.enqueueNDRange(m_queue,  
						  new long[]{},
		    			  new long[]{ half_size }, 
		    			  new long[]{ local_size }, 
		    			  new CLEvent[0]);
		}
		// 
	}
	
	
	void sweep(int[] src, int[] dst, int[] output)
	{
		for (int i = 0 ; i < src.length ; i ++)
		{
			output[i] = src[i] - dst[i];
		}
	}
	
	
	public static void main(String[] args) 
	{
		int[] output = new int[256];
		int[] samples  = new int[256];
		int[] sample = new int[] { 1,1,1,1,1,1,1,1};
		for (int i = 0 ;i < samples.length ; i ++)
			samples[i] = 1;
		
	}
	
	
}
