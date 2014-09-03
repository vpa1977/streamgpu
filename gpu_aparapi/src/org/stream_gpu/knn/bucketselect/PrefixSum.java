package org.stream_gpu.knn.bucketselect;


import com.amd.aparapi.Kernel;
import com.amd.aparapi.Kernel.Local;
import com.amd.aparapi.Range;
import com.amd.aparapi.device.Device;
import com.amd.aparapi.device.OpenCLDevice;
import com.amd.aparapi.opencl.*;


//------------------------------------------------------------
//Purpose :
//---------
//Prefix sum or prefix scan is an operation where each output element contains the sum of all input elements preceding it.
//
//Algorithm :
//-----------
//The parallel prefix sum has two principal parts, the reduce phase (also known as the up-sweep phase) and the down-sweep phase.
//
//In the up-sweep reduction phase we traverse the computation tree from bottom to top, computing partial sums.
//After this phase, the last element of the array contains the total sum.
//
//During the down-sweep phase, we traverse the tree from the root and use the partial sums to build the scan in place.
//
//Because the scan pictured is an exclusive sum, a zero is inserted into the last element before the start of the down-sweep phase.
//This zero is then propagated back to the first element.
//
//In our implementation, each compute unit loads and sums up two elements (for the deepest depth). Each subsequent depth during the up-sweep
//phase is processed by half of the compute units from the deeper level and the other way around for the down-sweep phase.
//
//In order to be able to scan large arrays, i.e. arrays that have many more elements than the maximum size of a work-group, the prefix sum has to be decomposed.
//Each work-group computes the prefix scan of its sub-range and outputs a single number representing the sum of all elements in its sub-range.
//The workgroup sums are scanned using exactly the same algorithm.
//When the number of work-group results reaches the size of a work-group, the process is reversed and the work-group sums are
//propagated to the sub-ranges, where each work-group adds the incoming sum to all its elements, thus producing the final scanned array.
//
//References :
//------------
//http://graphics.idav.ucdavis.edu/publications/print_pub?pub_id=1041
//
//To read :
//http://developer.apple.com/library/mac/#samplecode/OpenCL_Parallel_Prefix_Sum_Example/Listings/scan_kernel_cl.html#//apple_ref/doc/uid/DTS40008183-scan_kernel_cl-DontLinkElementID_5
//http://developer.apple.com/library/mac/#samplecode/OpenCL_Parallel_Reduction_Example/Listings/reduce_int4_kernel_cl.html
//------------------------------------------------------------

public class PrefixSum {
	
	public PrefixSum(Device dev)
	{
		kernel = ((OpenCLDevice)dev).bind(ScanBlockAnyLength.class);
	}
	
	public static void main(String[] args)
	{
		
		
		Device device = Device.firstGPU();
		PrefixSum sum = new PrefixSum(device);
		
		int[] array = new int[1024];
		for (int i = 0 ;i < array.length; i ++)
			array[i] = 1;
		
		array = sum.prefixSum(device, device.getMaxWorkGroupSize(), array);
		
		for (int i : array)
		{
			System.out.println(i);
		}
		
	}
	
	
	static int toMultipleOf(int N, int base) 
	{
		return (int) (Math.ceil((double)N / (double)base) * base);
	}
	
	ScanBlockAnyLength kernel;
	
	public int[] prefixSum(Device device , int workgroup_size,int[] dataSet)
	{
		int blockSize = dataSet.length / workgroup_size;
		int B = blockSize * workgroup_size;
		if ((dataSet.length  % workgroup_size) > 0)  
			blockSize++; 
		int localWorkSize = workgroup_size;
		int globalWorkSize = toMultipleOf(dataSet.length / blockSize, workgroup_size);
		
		
		Range range = device.createRange(globalWorkSize, localWorkSize);
		kernel.kernel__scan_block_anylength(range, new int[ workgroup_size ],
				dataSet,B, dataSet.length, blockSize);
		
		return dataSet;
	}
	
   @OpenCL.Resource("org/stream_gpu/knn/bucketselect/kernel_scan_block_anylength.cl") interface ScanBlockAnyLength extends
       OpenCL<ScanBlockAnyLength>{

    public ScanBlockAnyLength kernel__scan_block_anylength(//
          Range _range,//
          @Local(value = "localBuf") int[] localBuf,//
          @GlobalReadWrite(value = "dataSet") int[] dataSet, 
          @GlobalReadOnly( value ="B") int B,
          @GlobalReadOnly( value ="size") int size,
          @GlobalReadOnly( value ="passesCount") int passesCount        
          );
 }

	
}
