package org.stream_gpu.knn;

import org.bridj.Pointer;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLMem.MapFlags;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.LocalSize;

public class PrefixScan {
	
	private CLQueue m_queue;
	private CLContext m_context;
	private CLBuffer<Integer> m_digit_scan_output_buffer;
	private CLKernel m_digit_scan_kernel;
	private CLKernel m_prefix_sum_up_stage_0;
	private CLKernel m_prefix_sum_up_stage_n;
	private CLKernel m_prefix_sum_down_stage_n;
	private int m_group_size;
	private CLBuffer<Integer> m_sum_down_indices;

	public PrefixScan(CLContext context, CLQueue queue, int group_size) throws Throwable 
	{
		m_queue = queue;
		m_group_size = group_size;
		m_context = context;
		
		String kernel_source = KernelLoader.readKernel("radix.cl");
		
		CLProgram cl_program =  m_context.createProgram(kernel_source);
		m_digit_scan_kernel = cl_program.createKernel("scan_digit");
		m_prefix_sum_up_stage_0 = cl_program.createKernel("prefix_sum_group");
		m_prefix_sum_up_stage_n  = cl_program.createKernel("prefix_sum_global");
		m_prefix_sum_down_stage_n = cl_program.createKernel("prefix_sum_global_down");
		
		createSumDownIndices();
	}
	
	private void createSumDownIndices()
	{
		int[] indices = new int[ m_group_size * 2];
		int stride = 2;
	   while (stride <= m_group_size)
	   {
		   int pos = 0;
		   while (pos < 2*m_group_size) 
		   {
			   pos += stride;
			   if (pos-2 + stride/2 < indices.length)
				   indices[pos-2 + stride/2 ] = stride/2;
		   }
		   stride = stride * 2;
	   }
	   m_sum_down_indices = m_context.createIntBuffer(Usage.Input, indices.length);
	   Pointer<Integer> p = m_sum_down_indices.map(m_queue, MapFlags.Write, new CLEvent[0]);
	   p.setInts( indices );
	   m_sum_down_indices.unmap(m_queue, p, new CLEvent[0]);
	}
	
	
	
	CLBuffer<Integer> digitScanOutputBuffer()
	{
		return m_digit_scan_output_buffer;
	}
	
	/**
	 * Count digits
	 * @param input - vector of numbers where digits should be scanned, aligned to power of 2
	 * @return scan result - returns counts of each digit  - ones per workgroup, two's per workgroup etc =)
	 */
	public  CLBuffer<Integer> scan(CLBuffer<Integer> input,  int position)
	{
		int bitshift = position * 4;
		// 
		if (m_digit_scan_output_buffer == null || m_digit_scan_output_buffer.getElementCount() != 16 * (input.getElementCount() / m_group_size))
		{
			if (m_digit_scan_output_buffer != null)
				m_digit_scan_output_buffer.release();
			m_digit_scan_output_buffer = m_context.createIntBuffer(Usage.Output, 16 * (input.getElementCount() / m_group_size));
		}
		m_digit_scan_kernel.setArg(0, input);
		m_digit_scan_kernel.setArg(1, bitshift); //
		m_digit_scan_kernel.setArg(2, m_digit_scan_output_buffer); //
		m_digit_scan_kernel.setArg(3, LocalSize.ofByteArray(16 * m_group_size)); // - local counts
		m_digit_scan_kernel.enqueueNDRange( m_queue, null, new long[]{ input.getElementCount() } , new long[]{ m_group_size }, new CLEvent[0]);
		return m_digit_scan_output_buffer;
	}
	
	
	public void prefixSum(CLBuffer<Integer>  input)
	{
		prefixSumUp(input);
		prefixSumDown(input);
	}
	
	
	
	
	private void prefixSumUp(CLBuffer<Integer>  input) {
		
			long global_size = input.getElementCount()/2;
			long local_size = m_group_size;
			

			m_prefix_sum_up_stage_0.setArg(0,  input);
			m_prefix_sum_up_stage_0.setArg(1,  LocalSize.ofIntArray(global_size));
			m_prefix_sum_up_stage_n.setArg(0, input);
			m_prefix_sum_up_stage_n.setArg(1, LocalSize.ofIntArray(global_size));
			
			m_prefix_sum_up_stage_0.enqueueNDRange(m_queue, null,  new long[] {global_size} , new long[]{ local_size}, new CLEvent[0]);
		

		   long global_size_step;
		   local_size = m_group_size;
		   long num_groups = global_size / m_group_size;
		   int stride;

		   for (stride = 2*m_group_size, global_size_step= num_groups/2 ; stride <=  global_size ; stride =stride <<1, global_size_step = global_size_step >> 1) 
		   {
			   
			   m_prefix_sum_up_stage_n.setArg(2, stride);
			   
			   if (global_size_step < local_size) local_size = global_size_step;
			   
			   m_prefix_sum_up_stage_n.enqueueNDRange(m_queue, null, new long[] { global_size_step} , new long[] { local_size}, new CLEvent[0]);
		   }


	}



	private void prefixSumDown(CLBuffer<Integer>  input) {
		long local_size = m_group_size;
		long global_size = input.getElementCount();
		
		//(__global uint* src, __local uint* local_buf,const int global_stride, __global uint* indices)
		m_prefix_sum_down_stage_n.setArg(0, input);
		m_prefix_sum_down_stage_n.setArg(1, LocalSize.ofIntArray(2* local_size));
		m_prefix_sum_down_stage_n.setArg(2, m_sum_down_indices);

	    int stages =(int)(Math.log(global_size)/Math.log(2));
	    int stage_step =(int) ( Math.log(local_size)/Math.log(2));
	    for (long global_size_down = local_size, step = 1; global_size_down <= global_size; global_size_down = global_size_down << 1, ++step)
	    {
	 	   long current_stage = stages - step*stage_step;
	 	   long stride = stages < current_stage ? 1 : (long)Math.pow(2,current_stage);
	 	  m_prefix_sum_down_stage_n.setArg(3, (int)stride);
	 	  m_prefix_sum_down_stage_n.enqueueNDRange(m_queue,  null, new long[]{ global_size_down } , new long[] { local_size } , new CLEvent[0]);
	    }
	}



	public static void main(String[] args) throws Throwable 
	{
		int[] input = new int[1024];
		for (int i = 0; i < input.length; i++)
			input[i]= 1;
		
		CLContext context = JavaCL.createContext(null,
				JavaCL.listPlatforms()[0].listAllDevices(false)[0]);
		CLQueue queue = context.createDefaultQueue();
		int group_size = 256;
		PrefixScan scan = new PrefixScan( context, queue, 256);
		CLBuffer<Integer> buf = context.createIntBuffer(Usage.InputOutput, input.length);
		Pointer<Integer> ptr = buf.map(queue, MapFlags.Write, new CLEvent[0]);
		ptr.setInts(input);
		buf.unmap(queue, ptr, new CLEvent[0]);
		
		scan.prefixSum(buf);
		
		ptr = buf.map(queue, MapFlags.Read, new CLEvent[0]);
		input = ptr.getInts();
		
		buf.unmap(queue, ptr, new CLEvent[0]);
		
		System.out.println();
		
	}
	
	
}
