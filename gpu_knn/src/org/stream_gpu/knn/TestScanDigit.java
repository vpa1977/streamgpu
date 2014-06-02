package org.stream_gpu.knn;

import org.bridj.Pointer;
import org.junit.Test;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.LocalSize;
import com.nativelibs4java.opencl.CLMem.MapFlags;
import com.nativelibs4java.opencl.CLMem.Usage;

import junit.framework.TestCase;

public class TestScanDigit extends TestCase {
	
	/*@Test
	public void testScanDigit() throws Exception 
	{
		CLContext context = JavaCL.createContext(null,
				JavaCL.listPlatforms()[0].listAllDevices(false)[0]);

		String kernel = KernelLoader.readKernel("radix.cl");
		
		CLQueue queue = context.createDefaultProfilingQueue();
		
		CLKernel scan_kernel = context.createProgram(kernel).createKernel("scan_digit");
		
		int[] buffer = new int[] { 0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF, 0xF,
									   0xD, 0xE, 0xF, 0xF,0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF, 0xF,
									   0xA, 0xB, 0xF, 0xF,0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF, 0xF, 
									   0xC, 0x1, 0xF, 0xF,0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF, 0xF, 
									   0x0, 0x2, 0xF, 0xF,0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF, 0xF, 
									   0x2, 0x1,  0xF, 0xF,0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF, 0xF,0xF, 0xF, 0xF, 0xF};
		
		CLBuffer<Integer> input = context.createBuffer(Usage.Input, Integer.class, buffer.length);
		
		Pointer<Integer> pf = input.map(queue, MapFlags.Write, new CLEvent[0]);
		pf.setInts( buffer );
		input.unmap(queue, pf, new CLEvent[0]);
		
		int group_size = 16;
		
		CLBuffer<Integer> output = context.createIntBuffer(Usage.Output, 16 * (buffer.length / group_size));
//__kernel void scan_digit(__global float* src,  const int shift, 
//		 __global uint* global_counts, __local uint* local_counts) 
		scan_kernel.setArg(0, input);
		scan_kernel.setArg(1, 0); //
		scan_kernel.setArg(2, output); //
		scan_kernel.setArg(3, LocalSize.ofByteArray(16 * group_size)); // - local counts
		scan_kernel.enqueueNDRange(queue, null, new long[]{ buffer.length } , new long[]{ group_size }, new CLEvent[0]);
		queue.finish();
		
		Pointer<Integer> counts = output.map(queue, MapFlags.Read, null);
		int[] result = counts.getInts();
		output.unmap(queue, counts, null);
		
		
	}*/
	

	@Test
	public void testPrefixSum() throws Exception 
	{
		CLContext context = JavaCL.createContext(null,
				JavaCL.listPlatforms()[0].listAllDevices(false)[0]);

		String kernel = KernelLoader.readKernel("radix.cl");
		
		CLQueue queue = context.createDefaultProfilingQueue();
		
		CLKernel group_sum_kernel = context.createProgram(kernel).createKernel("prefix_sum_group");
		CLKernel global_sum_kernel = context.createProgram(kernel).createKernel("prefix_sum_global");
		CLKernel group_down_kernel  = context.createProgram(kernel).createKernel("prefix_sum_group_down");
		CLKernel global_down_kernel  = context.createProgram(kernel).createKernel("prefix_sum_global_down");
		
		
		int[] buffer = new int[1024];
		for (int i = 0 ; i < buffer.length ; i ++ )
			buffer[i]=1;
		
		CLBuffer<Integer> input = context.createBuffer(Usage.Input, Integer.class, buffer.length);
		Pointer<Integer> pf = input.map(queue, MapFlags.Write, new CLEvent[0]);
		pf.setInts( buffer );
		input.unmap(queue, pf, new CLEvent[0]);
		
		int group_size = 16;
		int global_size = buffer.length;
		
		group_sum_kernel.setArg(0, input);
		group_sum_kernel.setArg(1, LocalSize.ofIntArray(group_size));
		
		global_sum_kernel.setArg(0, input);
		global_sum_kernel.setArg(1, LocalSize.ofIntArray(group_size));
		
		global_down_kernel.setArg(0, input);
		global_down_kernel.setArg(1, LocalSize.ofIntArray(group_size));
		
		group_down_kernel.setArg(0, input);
		group_down_kernel.setArg(1, LocalSize.ofIntArray(group_size));
		
		
		global_sum_kernel.enqueueNDRange(queue, null, new long[]{ buffer.length } , new long[]{ group_size }, new CLEvent[0]);
		


		
		
//__kernel void scan_digit(__global float* src,  const int shift, 
//		 __global uint* global_counts, __local uint* local_counts) 
		
//		scan_kernel.enqueueNDRange(queue, null, new long[]{ buffer.length } , new long[]{ group_size }, new CLEvent[0]);
		//queue.finish();
		//
		//Pointer<Integer> counts = output.map(queue, MapFlags.Read, null);
		//int[] result = counts.getInts();
		//output.unmap(queue, counts, null);
		
		
	}
	
	/*public void testSequeunce()
	{
		int prev = 0;
		for (int stride = 1, step = 1; stride <= 128; stride = stride << 1, step ++ )
		{
			for (int i = 0 ;i < 256 ; i ++)
			{
				int value = i - prev;
				int pow = 1 << step;
				int mod = value & ( pow -1);
				if (mod == 0)
				{
					int offset = i;
					System.out.print("(" + offset + "-"+ (offset + stride) +")");
				}
			}
			prev += stride;
			System.out.println();
		}
	}*/
	
	@Test
	public void testDownSequence() 
	{
		
		int size = 16;
		int half_size = size /2;
		int prev = half_size -1;
		for (int stride = half_size/2, step = half_size; stride >0; stride = stride >> 1, step = step >> 1 )
		{
			for (int i = 0 ;i < size ; i ++)
			{
				int value = i - prev;
				if ( i != size -1)
				if (value >=0)	
				{
					int pow = step;
					int mod = value & ( pow -1);
					if (mod == 0)
					{
						int offset = i;
						System.out.print("(" + offset + "-"+ (offset + stride) +")");
					}
				}
			}
			prev -= stride;
			System.out.println();
		}

	}
}


