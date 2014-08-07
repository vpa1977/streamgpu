package org.stream_gpu.knn.aparapi;

import static org.junit.Assert.*;

import java.util.Random;










import javax.swing.plaf.basic.BasicInternalFrameTitlePane.MaximizeAction;

import org.junit.Test;

import com.amd.aparapi.Device;
import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.amd.aparapi.Kernel.Entry;
import com.amd.aparapi.OpenCL.GlobalReadWrite;
import com.amd.aparapi.OpenCL.GlobalWriteOnly;
import com.amd.aparapi.Range;
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

/** 
 * The APARAPI prototype of the prefix-scan kernel 
 * @author ÐÐÐ
 *
 */
public class PrefixScanAparapi {
	
	static int SIGNIFICANT_DIGITS = 7;
	static int DIGIT_VALUES = 16;
	
	
	private class MoveDigitKernel extends Kernel 
	{
		public long[] src;
		public long[] dst;
		public int shift;
		public long[] global_counts;
		@Local
		public long[] local_counts;
		public long[] range;
		@Override
		public void run() {
			int num_workgroups = getNumGroups(0);
			int group_size = getLocalSize(0)*2;
			int id = 2*getGlobalId(0);
			int local_index = 2*getLocalId(0);
			int mask =  (0xF << shift);
			int digit, digit1;
			int i; 
			int offset;
			// reset all local memory to 0
			for (digit=0; digit < DIGIT_VALUES; digit ++ ) 
			{
				offset = digit * group_size;
				local_counts[offset+local_index]=0;
				local_counts[offset+local_index+1]=0;
			}
			
			localBarrier();
				

			digit = (int)( (src[id] & mask) >> shift);
			
			// scan algorithm
			// scan order - digit at digit_pos for every element of the vector
			// local_counts - bit vectors local_count[0 ... workgroup_size] [ 0 .. NUM_DIGITS]
			offset = digit * group_size + local_index;
			local_counts[offset]=1;

			digit1 =  (int)( (src[id+1] & mask) >> shift );
			offset = digit1 * group_size + local_index+1;
			local_counts[offset]=1;

			for (int cur_digit=0; cur_digit < DIGIT_VALUES; cur_digit ++ ) 
			{
				int global_offset = cur_digit * group_size;
		        int tid = getLocalId(0);
		//  see https://code.google.com/p/clpp/source/browse/trunk/src/clpp/clppScan_Default.cl?r=126        
			    offset = 1;
				// Build the sum in place up the tree
				for(i = group_size>>1; i > 0; i >>=1)
				 {
					localBarrier();
				    if(tid<i)
				    {
				            int ai = offset*(2*tid + 1) - 1;
				            int bi = offset*(2*tid + 2) - 1;
				            ////printf("grp %d sum up %d -> %d\n", get_group_id(0) , ai, bi);
				            local_counts[bi+global_offset] += local_counts[ai+global_offset];
				    }
				    offset *= 2;
		        }
				
			    // scan back down the tree
			    // Clear the last element
				local_counts[global_offset+ group_size - 1] = 0;
				localBarrier();
				
				 
				// traverse down the tree building the scan in the place
			    for(i = 1; i < group_size ; i *= 2)
			    {
				    offset >>=1;
				    localBarrier();
				                
				    if(tid < i)
				    {
				            int ai = offset*(2*tid + 1) - 1;
				            int bi = offset*(2*tid + 2) - 1;
				                        
				            float t = local_counts[ai+global_offset];
				            local_counts[ai+global_offset] = local_counts[bi+global_offset];
				            local_counts[bi+global_offset] += t;
				    }
			     }
			}

			localBarrier();
			
			
			offset = (int) (global_counts[(int) digit * num_workgroups + getGroupId(0)] - local_counts[(int) digit * group_size + local_index] -1);
			dst[ offset ] = src[id];
			offset = (int)(global_counts[digit1 * num_workgroups + getGroupId(0)] - local_counts[ digit1 * group_size + local_index+1]-1);
			dst[ offset ] = src[id+1];
			
		}
		
	}
	
	
	private class DigitScanKernel extends Util 
	{
		
		public long[] src;  // source ints
		public int shift;  // bit shift
		public long[] global_counts; // counts of digits 0=> [0 ... workgroup_count] 1=>[ workgroup_count +1... workgroup_count + workgroup_count] etc
		public static final int NUM_DIGITS = 16;
		


		@Override
		public void run() {
			
			int workgroup = getGroupId(0);
			int num_workgroups = getNumGroups(0);
			int workgroup_size = 2*getLocalSize(0);
			int local_index = 2*getLocalId(0);
			int global_index = 2*getGlobalId(0);
			int mask =  (0xF << shift);
			long digit;
			int i; 
			int offset;
			// reset all local memory to 0
			for (digit=0; digit < NUM_DIGITS; digit ++ ) 
			{
				offset = (int)(digit * workgroup_size);
				local_counts[offset+local_index]=0;
				local_counts[offset+local_index+1]=0;
			}
			
			localBarrier();

			digit =  (src[global_index] & mask) >> shift;
			// scan algorithm
			// scan order - digit at digit_pos for every element of the vector
			// local_counts - bit vectors local_count[0 ... workgroup_size] [ 0 .. NUM_DIGITS]
			offset = (int)(digit * workgroup_size + local_index);
			local_counts[offset]=1;
			
			digit =  (src[global_index+1] & mask) >> shift;
			// scan algorithm
			// scan order - digit at digit_pos for every element of the vector
			// local_counts - bit vectors local_count[0 ... workgroup_size] [ 0 .. NUM_DIGITS]
			offset = (int)(digit * workgroup_size + local_index+1);
			local_counts[offset]=1;
			 localBarrier();
			 
			
			// local_counts reduced into global_counts
			// global_counts  offset vector for each digit/ workgroup [0 ... num_workgroups] [ 0.. NUM_DIGITS]
		 	 for (digit = 0 ;digit < NUM_DIGITS ; digit ++)
			 {
				offset =(int)(digit * workgroup_size);
				
				reduce(workgroup_size, local_index, offset);
			}
			localBarrier();
			if (local_index == 0 ) 
			{
				for (digit =0; digit < NUM_DIGITS; digit ++ ) 
				{
					global_counts[(int)(num_workgroups * digit + workgroup)] = local_counts[(int)(digit * workgroup_size)];
				}
			}

		}

		
	}
	
	
	private int m_max_step;
	private int m_max_down;
	
	
	
	public PrefixScanAparapi(){}
	
	public void sort(long[] values)
	{
		int global_size = values.length;
		
		int[] up_indices  = new int[global_size];
		int[] down_indices = new int[global_size];
		createSumIndices(up_indices, down_indices);
	}

	private void createSumIndices(int[] up_indices, int[] down_indices)
	{

		int size = up_indices.length;
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

	    m_max_step = max; 
	}
	
	@Test 
	public void testPrefixSum()
	{
		Device gpu = Device.firstGPU();
		int global_size = 4096;
		int local_size = 256;
		final long[] prefix_sum = new long[global_size];
		int[] up_indices  = new int[global_size];
		int[] down_indices = new int[global_size];
		createSumIndices(up_indices, down_indices);
		
		for (int i = 0 ; i < global_size ; i ++ )
		{
			prefix_sum[i] = 1;
		}
		
		PrefixSumUpKernel sum_up_kernel = new PrefixSumUpKernel();
		
		sum_up_kernel.local_buf = new long[ local_size ];
		sum_up_kernel.max_up = m_max_step;
		sum_up_kernel.up_indices = up_indices;
		sum_up_kernel.src = prefix_sum;
		Range up_range = gpu.createRange(global_size/2, local_size/2);
		sum_up_kernel.setExecutionMode(EXECUTION_MODE.JTP);
		sum_up_kernel.execute(up_range);

		PrefixSumDownKernel sum_down_kernel = new PrefixSumDownKernel();
		sum_down_kernel.src = prefix_sum;
		
		printArray(prefix_sum);
		
		for (int step = m_max_down-1, size = 2; step > 0 ; step --, size = size *2 )
		{
			sum_down_kernel.offset = (1 << step) - 1;
			sum_down_kernel.stride = (1 << (step-1));
			int cur_global_size = size -1;
			Range down_range = gpu.createRange(cur_global_size);
			sum_down_kernel.execute(down_range);
		}
		long[] result = sum_down_kernel.src;
		for (int i = 1 ;i < result.length ; i ++ )
			assertEquals(result[i], result[i-1]+1);
	}
	
	

	@Test
	public void testScanDigit()
	{
		Device gpu = Device.firstGPU();
		
		int global_size = 4096;
		int local_size = 256;
		
		final long[] digits_to_scan = new long[global_size];
		for (int i = 0 ;i < digits_to_scan.length ; i ++)
			digits_to_scan[i] = 1 + i % (0xF);
		
		DigitScanKernel scan_kernel = new DigitScanKernel();
		scan_kernel.src = digits_to_scan;
		scan_kernel.shift = 0;
		scan_kernel.local_counts = new int[local_size * DigitScanKernel.NUM_DIGITS];
		scan_kernel.global_counts = new long[DigitScanKernel.NUM_DIGITS * global_size/local_size];
		Range rng = gpu.createRange(global_size/2, local_size/2);
		
		//scan_kernel.setExecutionMode(EXECUTION_MODE.JTP);
		scan_kernel.execute(rng);
		long[] result = scan_kernel.global_counts;
		for (int i = 0 ;i < result.length ; i ++ )
			System.out.print(result[i] +" ");
		
		System.out.println(scan_kernel.global_counts[0]);
		
		
	}
	
	private void printArray(long[] array)
	{
		for (int i = 0 ; i < array.length ; i ++ )
		{
			System.out.print(array[i] +" ");
		}
		System.out.println();
	}

	
}
