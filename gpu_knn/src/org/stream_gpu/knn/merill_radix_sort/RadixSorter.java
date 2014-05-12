package org.stream_gpu.knn.merill_radix_sort;

public class RadixSorter {
	
	
	public void select(int[] f, int k)
	{
		int[] buffer = new int[f.length];
		
		int num_passes = max_pass(f);
		int[][] spine = new int[num_passes][0xF];
		int[][] spine_offset = new int[num_passes][0xF];
		
		for (int pass = num_passes ; pass >=0 ; pass--) {
			clear_spine(spine[pass], spine_offset[pass]);
			count_digits( spine[pass], f , pass);
			make_offsets( spine[pass]);
			reorder(f , buffer, spine, spine_offset, pass );
		}
	}
	
	private void reorder( int[] src, int[] dst, int[][] spine, int[][] spine_offset, int pass)
	{
		
	}

	private void make_offsets(int[] spine) {
		for (int i = 0 ;i < spine.length-1 ; i ++ )
			spine[i+1] += spine[i];
		
	}



	private void count_digits(int[] spine, int[] f, int pass) {
		int shift = pass *4;
		for (int i = 0 ;i < f.length ; i ++ )
		{
			int value = (f[i] & ( 0xF << shift)) >> shift;
			spine[value]++;
		}
	}
	
	private int max_pass(int[] f)
	{
		int max = f[0];
		for (int i = 1 ; i < f.length ; i ++)
			if (max < f[i])
				max = f[i];
		return (int)(Math.log(max)/Math.log(16));
	}

	private void clear_spine(int[] spine, int[] spine_offset) {
		for (int i = 0 ;i < spine.length ; i ++)
		{
			spine[i] = 0;
			spine_offset[i] = 0;
		}
	}
	
	
	public static void main(String[] args)
	{
		
		int [] values = new int[] { 0xFF, 0x1233, 0x44, 0x132211 };
		new RadixSorter().select(values ,2 );
	}
	
	
}
