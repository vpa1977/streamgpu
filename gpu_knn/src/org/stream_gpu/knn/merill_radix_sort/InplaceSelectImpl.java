package org.stream_gpu.knn.merill_radix_sort;

/**
 * To be used with k < 1<<21 => 2,097,152 
 */
public class InplaceSelectImpl {

	
	

	public void work(int global_id, int local_id, float[] input, int  total_size, int k, int[] spine, boolean[] temp_from_alt	)
	{
		
	}
	
	
  class compare_radix {
	  public compare_radix(int radix, int bit) { this.radix = radix; this.bit = bit; }
	    int operator(int y) {
	      if(y != 0xFFFFFFFF && y !=0x00000000){
	        int temp = (y >> bit) & 0x0000000F;
	        if(temp < radix){
	          y = 0x00000000;
	        }
	        else if( temp > radix){
	          y = 0xFFFFFFFF;
	        }
	      }
	      return y;
	    }
	    int radix, bit;
	  };
	
	
	
	
	public void run()
	{
		int k = 10;
		int size = 100;
		int wg_size = 8;
		
		float[] input = new float[size];
		float[] output = new float[size];
		
		int[] spine = createSpine(size,k);
		
		for (int i=0; i < size; i ++ )
		{
			
			work( i, i % wg_size, input , size, k,  spine, new boolean[2]);
		}
	}

	private int[] createSpine(int size, int k) {
		// TODO Auto-generated method stub
		return null;
	}


	private float[] createStorage(int size, int k) {
		return new float[size];
	}


	public static void main(String[] args){
		new InplaceSelectImpl().run();
	}
}
