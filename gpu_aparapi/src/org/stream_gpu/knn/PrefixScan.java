package org.stream_gpu.knn;

import java.util.Random;

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
	
	static int SIGNIFICANT_DIGITS = 7;
	static int DIGIT_VALUES = 16;
	
	private CLQueue m_queue;
	private CLContext m_context;
	private CLBuffer<Integer> m_digit_scan_output_buffer;
	private CLKernel m_scan_and_move_digit;
	private CLKernel m_digit_scan_kernel;
	private CLKernel m_prefix_sum_up;
	private CLKernel m_prefix_sum_down;
	
	private int m_group_size;
	private int m_max_step;
	private int m_max_down;
	private int m_global_size;
	private int m_half_local_size;
	private int m_scan_group_size;
	
	private CLBuffer<Integer> m_sum_down_indices;
	private CLBuffer<Integer> m_status;
	
	private CLBuffer<Integer> m_sum_up_indices;
	private CLKernel m_adjust_k;

	public PrefixScan(CLContext context, CLQueue queue, int group_size, int global_size) throws Throwable 
	{
		m_queue = queue;
		m_group_size = group_size;
		m_scan_group_size = group_size;
		m_context = context;
		
		String kernel_source = KernelLoader.readKernel("radix.cl");
		
		CLProgram cl_program =  m_context.createProgram(kernel_source);
		m_digit_scan_kernel = cl_program.createKernel("scan_digit");
		m_prefix_sum_up = cl_program.createKernel("prefix_sum_up");
		m_prefix_sum_down = cl_program.createKernel("prefix_sum_down");
		m_scan_and_move_digit = cl_program.createKernel("scan_and_move_digit");
		m_adjust_k = cl_program.createKernel("adjust_k");
		//m_prefix_sum_up_stage_n  = cl_program.createKernel("sum_up");
		//m_prefix_sum_down_stage_n = cl_program.createKernel("sum_down");
		
		m_global_size = global_size;
		m_group_size = Math.min(group_size, global_size);
		m_half_local_size = Math.min( m_global_size/2 , m_group_size);
		createSumIndices( global_size );
		
		m_prefix_sum_up.setArg(1,  LocalSize.ofIntArray(m_group_size));
		m_prefix_sum_up.setArg(2,  m_sum_up_indices);
		m_prefix_sum_up.setArg(3,  m_max_step);
		
		m_status = m_context.createIntBuffer(Usage.Input, 3);
	}
	
	public void setupStatus()
	{
		Pointer<Integer> ptr = m_status.map(m_queue, MapFlags.Write, new CLEvent[0]);
		ptr.setInts(new int[]{0,0});
		m_status.unmap(m_queue, ptr, new CLEvent[0]);
	}
	
	public void prefixSum(CLBuffer<Integer> buf) {
		m_prefix_sum_up.setArg(0,  buf);
		m_prefix_sum_up.enqueueNDRange(m_queue, null,  new long[] {m_global_size/2} , new long[]{ m_half_local_size}, new CLEvent[0]);
		  
		for (int step = m_max_down ; step > 0 ; step -- )
		{
			m_prefix_sum_down.setArg(0,  buf);
			m_prefix_sum_down.setArg(1,  step);
			int cur_global_size = (int)(m_global_size / (1 << (step)));
			int local =  (int)Math.min(cur_global_size,  m_half_local_size);
			m_prefix_sum_down.enqueueNDRange(m_queue, new long[] {1},  new long[] {cur_global_size} , new long[]{local}, new CLEvent[0]);
		}
		
		
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
				int down_sum_index = up_sum_index;
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
		
		down_indices[down_indices.length -1 ] = 0;

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
	

	CLBuffer<Integer> digitScanOutputBuffer()
	{
		return m_digit_scan_output_buffer;
	}
	
	/**
	 * Count digits
	 * @param input - vector of numbers where digits should be scanned, aligned to power of 2 and greater that m_group_size
	 * @return scan result - returns counts of each digit  - ones per workgroup, two's per workgroup etc =)
	 */
	public  CLBuffer<Integer> scan(CLBuffer<Integer> input,  int position)
	{
		if (input.getElementCount() < m_group_size) 
			throw new IllegalArgumentException("element count is less than group size");
		int bitshift = position * 4;
		// 
		if (m_digit_scan_output_buffer == null)
		{
			System.out.println("Created position buffer "+ (16 * (input.getElementCount() / (2*m_group_size))));
			m_digit_scan_output_buffer = m_context.createIntBuffer(Usage.Output, 16 * (input.getElementCount() / (2*m_group_size) ));
		}
		m_digit_scan_kernel.setArg(0, input);
		m_digit_scan_kernel.setArg(1, bitshift); //
		m_digit_scan_kernel.setArg(2, m_digit_scan_output_buffer); //
		m_digit_scan_kernel.setArg(3, LocalSize.ofByteArray(16 * Math.min(m_scan_group_size, input.getElementCount()/2 ) *2 )); // - local counts
		//m_digit_scan_kernel.enqueueNDRange( m_queue, null, new long[]{ input.getElementCount() } , new long[]{ m_group_size }, new CLEvent[0]);
		m_digit_scan_kernel.enqueueNDRange( m_queue, null,new long[]{ input.getElementCount()/2 } , new long[]{Math.min(m_scan_group_size, input.getElementCount()/2 ) }, new CLEvent[0]);
		return m_digit_scan_output_buffer;
	}
	
	public void checkBuffer(CLBuffer<Integer> buf)
	{
		Pointer<Integer> ptr = buf.map(m_queue, MapFlags.Read, new CLEvent[0]);
		int[] input = ptr.getInts();
		for (int i = 0 ;i < input.length ; i ++ )
			System.out.print(" " + input[i]);
		System.out.println();
		buf.unmap(m_queue, ptr, new CLEvent[0]);
	}
	
	public void isPrefixSum(CLBuffer<Integer> buf) throws Exception
	{
		Pointer<Integer> ptr = buf.map(m_queue, MapFlags.Read, new CLEvent[0]);
		int[] input = ptr.getInts();
		buf.unmap(m_queue, ptr, new CLEvent[0]);
		for (int i = 1 ; i < input.length ; i ++ )
		{
			if (input[i] - input[i-1] != 1)
				throw new Exception("oh damn");
		}
	}

	
	private static void testPrefixSum() throws Throwable
	{
		int group_size = 256;
		int[] input = new int[4096];
		for (int i = 0; i < input.length; i++)
			input[i]= 1;
		
		
		
		CLContext context = JavaCL.createContext(null,
				JavaCL.listPlatforms()[0].listAllDevices(false)[0]);
		CLQueue queue = context.createDefaultQueue();
		
		
		PrefixScan scan = new PrefixScan( context, queue, group_size, input.length);
		CLBuffer<Integer> buf = context.createIntBuffer(Usage.InputOutput, input.length);
		Pointer<Integer> ptr = buf.map(queue, MapFlags.Write, new CLEvent[0]);
		ptr.setInts(input);
		buf.unmap(queue, ptr, new CLEvent[0]);
		
		scan.prefixSum(buf);
		scan.isPrefixSum(buf);
		

		System.out.println();
		
	}



	public static void main(String[] args) throws Throwable 
	{
		
			testPrefixSum();
		Random rnd = new Random();
		int group_size = 256;
		int[] input = new int[4096];
		for (int i = 0; i < input.length; i++)
			input[i]= input.length - i;
		
		CLContext context = JavaCL.createContext(null,
				JavaCL.listPlatforms()[0].listAllDevices(false)[0]);
		CLQueue queue = context.createDefaultQueue();
		
		
		
		PrefixScan scan = new PrefixScan( context, queue, group_size, (input.length/group_size) * DIGIT_VALUES);
		
		CLBuffer<Integer> swap = context.createIntBuffer(Usage.InputOutput, input.length);
		CLBuffer<Integer> buf = context.createIntBuffer(Usage.InputOutput, input.length);
		CLBuffer<Integer> tmp;
		
		Pointer<Integer> ptr = buf.map(queue, MapFlags.Write, new CLEvent[0]);
		ptr.setInts(input);
		buf.unmap(queue, ptr, new CLEvent[0]);
		
		long start = System.currentTimeMillis();
		int k = 3;
		scan.setupStatus();
		for (int pos =0 ; pos >=0 ; --pos)
		{
			CLBuffer<Integer> positions = scan.scan(buf, pos );
			scan.checkBuffer(positions);
			scan.prefixSum(positions);
			scan.checkBuffer(positions);
			scan.adjust_k(k, positions, buf);
			scan.scanAndSwap( buf,swap, pos, positions, k);
			
			tmp = buf;
			buf = swap;
			swap = tmp;
		}
		//scan.checkBuffer(buf);
		
		
		long end = System.currentTimeMillis();
		

		System.out.println( (end-start)); // 284
		
	}

	
	private void adjust_k(int k, CLBuffer<Integer> positions, CLBuffer<Integer> input) {
		/*m_adjust_k.setArg(0, positions);
		m_adjust_k.setArg(1, k);
		m_adjust_k.setArg(2, (int)(input.getElementCount() / (2*m_group_size)));
		m_adjust_k.setArg(3, m_status);
		
		m_adjust_k.enqueueNDRange(m_queue, null, new long[] {16},new long[]{16}, new CLEvent[0]); 
		checkBuffer(m_status);
		*/
		Pointer<Integer> ptr = positions.map(m_queue, MapFlags.Read, new CLEvent[0]);
		int[] pos = ptr.getInts();
		positions.unmap(m_queue, ptr, new CLEvent[0]);
		
	}

	private CLBuffer<Integer> scanAndSwap(CLBuffer<Integer> input, CLBuffer<Integer> output, int position, CLBuffer<Integer> positions, int k) {
		if (input.getElementCount() < m_group_size) 
			throw new IllegalArgumentException("element count is less than group size");

			int bitshift = position * 4;
			m_scan_and_move_digit.setArg(0, input);
			m_scan_and_move_digit.setArg(1, output);
			m_scan_and_move_digit.setArg(2, bitshift); //
			m_scan_and_move_digit.setArg(3, positions); //
			m_scan_and_move_digit.setArg(4, LocalSize.ofByteArray(16 * Math.min(m_scan_group_size, input.getElementCount()/2 )*2 )); // - local counts
			m_scan_and_move_digit.setArg(5, m_status);
			m_scan_and_move_digit.enqueueNDRange( m_queue,null, new long[]{ input.getElementCount()/2 } , new long[]{Math.min(m_scan_group_size, input.getElementCount()/2 ) }, new CLEvent[0]);
			m_queue.finish();
		
		return m_digit_scan_output_buffer;
	}

	

	
	
}
