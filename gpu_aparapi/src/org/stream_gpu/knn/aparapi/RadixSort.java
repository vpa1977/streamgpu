package org.stream_gpu.knn.aparapi;



import com.amd.aparapi.OpenCL;
import com.amd.aparapi.Range;
import com.amd.aparapi.OpenCL.GlobalReadOnly;
import com.amd.aparapi.OpenCL.GlobalReadWrite;
import com.amd.aparapi.OpenCL.Local;

public class RadixSort {

	
	public int[] radixSort(int[] keys, int[] values)
	{
		return keys;
	}
	
	
	
	@OpenCL.Resource("org/stream_gpu/knn/aparapi/radix_sort_gpu_gems.cl") interface RadixSortKernel extends
     OpenCL<RadixSortKernel>{

  public RadixSort kernel__radixLocalSort(//
        Range _range,//
        @GlobalReadWrite(value = "data") int[] dataSet, 
        @GlobalReadOnly( value ="bitOffset") int bitOffset,
        @GlobalReadOnly( value ="N") int N);
  public RadixSort kernel__localHistogram(
	        @GlobalReadWrite(value = "data") int[] dataSet, 
	        @GlobalReadOnly( value ="bitOffset") int bitOffset,
	        @GlobalReadWrite(value = "radixCount") int[] radixCount,
	        @GlobalReadWrite(value = "radixOffsets") int[] radixOffsets,
	        @GlobalReadOnly( value ="N") int N);
  
  public RadixSort kernel__radixPermute(
		  	@GlobalReadWrite(value = "dataIn") int[] dataSet,
		  	@GlobalReadWrite(value = "dataOut") int[] dataOut,
		  	@GlobalReadOnly(value = "histSum") int[] histSum,
		  	@GlobalReadOnly(value = "blockHists") int[] blockHists,
		  	@GlobalReadOnly(value = "bitOffset") int bitOffset,
		  	@GlobalReadOnly(value = "N") int N,
		  	@GlobalReadOnly(value = "numBlocks") int numBlocks);
  
	}
 
}
