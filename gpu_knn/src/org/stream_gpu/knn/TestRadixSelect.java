package org.stream_gpu.knn;

import org.junit.Test;

import junit.framework.TestCase;


public class TestRadixSelect extends TestCase {
	
	void fill( int[] arr, int start)
	{
		for (int i = 0; i <  arr.length ; i ++)
			arr[i] = 300;
	}

	@Test	
	public void testSelect()
	{
		int src[] = new int[32];
		
		fill(src, 1000);
		src[1]=1;
		src[2]=1;
		src[30]=1;
		radixSelect(src, 15);
		
		
	}
	
	private void printArray(int[] arr)
	{
		for (int i : arr)
		{
			System.out.print(" " + Integer.toHexString(i));
		}
		System.out.println();	
		
	}

	private int[] radixSelect(int[] src, int k) {
		// scan
		int workgroup_size = 4;
		int num_grps = src.length / workgroup_size;
		
		int[] selection = new int[src.length];
		
		int position = 0;
		int[] dst = new int[src.length];
		
		
		
		for (int  current_digit= SIGNIFICANT_DIGITS ; current_digit >=0 ; current_digit --)
		{
			int[] global_counts = new int[DIGIT_VALUES*workgroup_size];
			
			for (int grp = 0 ; grp < num_grps; grp ++ )
			{
				int[] local_counts = new int[DIGIT_VALUES];
				for (int local_index = 0;local_index < workgroup_size ; local_index ++)
					scan_digits(local_index,grp ,workgroup_size, src, current_digit, global_counts,  local_counts);
			}
			
			prefix_sum( global_counts);
			
			int digit = find_digit(global_counts, k - position);
			// move digits
			if (global_counts[0] != global_counts[global_counts.length-1]) // shortcut
			{
				move_digits(src, current_digit, selection, position,  digit);
				if (digit > 0)
					position = global_counts[digit-1] + position;
			}
		}
		if (position==0)
			return src;
		return selection;
	}
	
	private void move_digits(int[] src,int digit_pos, int[] selection, int start_pos, int digit)
	{
		long shift = 4 * digit_pos;
		int mask =  (0xF << shift);
		for (int i = 0 ; i < src.length ; i ++ )
		{
			int value = (src[i] & mask) >> shift;
			if (value < digit) {
				selection[start_pos++]  = src[i];
				src[i] = Integer.MAX_VALUE;
			}
			else
			if (value > digit)
				src[i] = Integer.MAX_VALUE;
		
			if (digit_pos ==0 && value == digit)
				selection[start_pos++]  = src[i];
		}
	}
	
	private int find_digit( int[] global_counts, int k)
	{
		int next_stage  = -1;
		for (int i =0 ;i < global_counts.length ; i ++)
		{
			if (k < global_counts[i]) // shortcut for k == global_counts[i]
			{
				next_stage = i;
				break;
			}
		}
		return next_stage;
	}

	/**
	 * simplistic scan digits.
	 * @param src
	 * @param digit_pos
	 * @param global_counts
	 */
	private void scan_digits(int local_index, int workgroup, int workgroup_size, int[] src, int digit_pos, int[] global_counts ,  int[] local_counts)
	{
		long shift = 4 * digit_pos;
		int mask =  (0xF << shift);
		int value = (src[local_index] & mask) >> shift;
		local_counts[(int)value] +=1;
		// barrier
		if (local_index == workgroup_size-1) // change to 0 in the kernel
		{
			for (int i = 0 ;i < DIGIT_VALUES ; i ++) // unroll loop in the kernel
				global_counts[workgroup * DIGIT_VALUES +i] = local_counts[ i ];
		}
	}
	
	private void prefix_sum( int[] counts)
	{
		for (int i =  1 ; i < counts.length ; i ++)
			counts[i] += counts[i-1];
	}
	
	static int SIGNIFICANT_DIGITS = 7;
	static int DIGIT_VALUES = 16;
}
