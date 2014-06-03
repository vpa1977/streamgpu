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
	private CLKernel m_prefix_sum_up;
	private CLKernel m_prefix_sum_down;
	
	private int m_group_size;
	private int m_max_step;
	private int m_max_down;
	
	private CLBuffer<Integer> m_sum_down_indices;
	
	private CLBuffer<Integer> m_sum_up_indices;

	public PrefixScan(CLContext context, CLQueue queue, int group_size, int global_size) throws Throwable 
	{
		m_queue = queue;
		m_group_size = group_size;
		m_context = context;
		
		String kernel_source = KernelLoader.readKernel("radix.cl");
		
		CLProgram cl_program =  m_context.createProgram(kernel_source);
		m_digit_scan_kernel = cl_program.createKernel("scan_digit");
		m_prefix_sum_up = cl_program.createKernel("prefix_sum_up");
		m_prefix_sum_down = cl_program.createKernel("prefix_sum_down");
		//m_prefix_sum_up_stage_n  = cl_program.createKernel("sum_up");
		//m_prefix_sum_down_stage_n = cl_program.createKernel("sum_down");
		
		
		createSumIndices( global_size );
	}
	
	

	private void createSumIndices(int size)
	{
		
		int[] up_indices = new int[size];
		int[] down_indices = new int[size];
		int stride = 2;
		int max = 0;
		int prev = 0;
		while (stride <=  size)
		{
			max +=1;
			for (int pos = prev; pos < size; pos += stride)
			{
				int up_sum_index = pos + stride/2;
				int down_sum_index = up_sum_index + stride/2;
				up_indices[up_sum_index]+=1;
				if (down_sum_index < down_indices.length )
				{
					down_indices[down_sum_index ] = max;
					m_max_down = max;
				}
			}
			prev += stride/2;
			stride = stride *2;
			
		}

		 m_sum_up_indices = m_context.createIntBuffer(Usage.Input, up_indices.length);
	     Pointer<Integer> p = m_sum_up_indices.map(m_queue, MapFlags.Write, new CLEvent[0]);
	     p.setInts( up_indices );
	     m_sum_up_indices.unmap(m_queue, p, new CLEvent[0]);
		 
		 m_sum_down_indices = m_context.createIntBuffer(Usage.Input, down_indices.length);
	     p  = m_sum_down_indices.map(m_queue, MapFlags.Write, new CLEvent[0]);
	     p.setInts( down_indices );
	     m_sum_down_indices.unmap(m_queue, p, new CLEvent[0]);
	     m_max_step = max; 
	}
	
	/*
	private void createSumDownIndices(int size)
	{
	   int half_size = size/2;				
	   int[] indices = new int[ size];
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
	
	*/
	
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
	
	public void checkBuffer(CLBuffer<Integer> buf)
	{
		Pointer<Integer> ptr = buf.map(m_queue, MapFlags.Read, new CLEvent[0]);
		int[] input = ptr.getInts();
		buf.unmap(m_queue, ptr, new CLEvent[0]);
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
		
		PrefixScan scan = new PrefixScan( context, queue, group_size, input.length);
		CLBuffer<Integer> buf = context.createIntBuffer(Usage.InputOutput, input.length);
		Pointer<Integer> ptr = buf.map(queue, MapFlags.Write, new CLEvent[0]);
		ptr.setInts(input);
		buf.unmap(queue, ptr, new CLEvent[0]);
		
		scan.prefixSum(buf);
		scan.checkBuffer(buf);
		

		System.out.println();
		
	}


	public void prefixSum(CLBuffer<Integer> buf) {
		long  global_size = buf.getElementCount();
		long local_size = Math.min(m_group_size, global_size);
		m_prefix_sum_up.setArg(0,  buf);
		m_prefix_sum_up.setArg(1,  LocalSize.ofIntArray(local_size));
		m_prefix_sum_up.setArg(2,  m_sum_up_indices);
		m_prefix_sum_up.setArg(3,  m_max_step);
		
		long up_local_size = Math.min( global_size/2 , local_size);
		m_prefix_sum_up.enqueueNDRange(m_queue, null,  new long[] {global_size/2} , new long[]{ up_local_size}, new CLEvent[0]);
		m_prefix_sum_down.setArg(0,  buf);
		m_prefix_sum_down.setArg(1,  LocalSize.ofIntArray(local_size));
		m_prefix_sum_down.setArg(2,  m_sum_down_indices);
		m_prefix_sum_down.setArg(3,  m_max_down);
		
		m_prefix_sum_down.enqueueNDRange(m_queue, null,  new long[] {global_size} , new long[]{ local_size}, new CLEvent[0]);
	}
	
	
}
