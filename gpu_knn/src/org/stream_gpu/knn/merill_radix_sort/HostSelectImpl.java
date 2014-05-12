package org.stream_gpu.knn.merill_radix_sort;

public class HostSelectImpl {
	
	
	public void work(int global_id, int local_id, float[] input, float[] output)
	{
		int val = Float.floatToIntBits(input[global_id]);
		
	}
	
	
	public void run()
	{
		int size = 100;
		int wg_size = 8;
		float[] input = new float[size];
		float[] output = new float[size];
		for (int i=0; i < size; i ++ )
		{
			work( i, i % wg_size, input, output);
		}
	}

	public static void main(String[] args){
		new HostSelectImpl().run();
	}
}
