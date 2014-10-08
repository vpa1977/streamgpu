package org.stream_gpu.knn;

import com.amd.aparapi.Range;
import com.amd.aparapi.device.Device;
import com.amd.aparapi.device.OpenCLDevice;
import com.amd.aparapi.opencl.OpenCL;



public class BitonicSortAparapi {
	
	
	@OpenCL.Resource("org/stream_gpu/knn/sort.cl")
	interface BitonicSortAparapiKernel extends OpenCL<BitonicSortAparapiKernel>
	{
		 public BitonicSortAparapiKernel bsort_init(//
		          Range _range,//
		          @GlobalWriteOnly(value = "g_data") float[] g_data,//
		          @Local(value = "l_data") float[] l_data, 
		          @GlobalWriteOnly( value ="g_indices") int[] g_indices,
		          @Local( value ="l_indices") int[] l_indices   
		          );
		 
		 public BitonicSortAparapiKernel bsort_stage_0(//
		          Range _range,//
		          @GlobalWriteOnly(value = "g_data") float[] g_data,//
		          @Local(value = "l_data") float[] l_data,
		          @Constant(value="high_stage") int high_stage,
		          @GlobalWriteOnly( value ="g_indices") int[] g_indices,
		          @Local( value ="l_indices") int[] l_indices   
		          );

		 public BitonicSortAparapiKernel bsort_stage_n(//
		          Range _range,//
		          @GlobalWriteOnly(value = "g_data") float[] g_data,//
		          @Local(value = "l_data") float[] l_data,
		          @Constant(value="stage") int stage,
		          @Constant(value="high_stage") int high_stage,
		          @GlobalWriteOnly( value ="g_indices") int[] g_indices,
		          @Local( value ="l_indices") int[] l_indices   
		          );

		 
		 public BitonicSortAparapiKernel bsort_merge(//
		          Range _range,//
		          @GlobalWriteOnly(value = "g_data") float[] g_data,//
		          @Local(value = "l_data") float[] l_data,
		          @Constant(value="stage") int stage,
		          @Constant(value="dir") int dir,
		          @GlobalWriteOnly( value ="g_indices") int[] g_indices,
		          @Local( value ="l_indices") int[] l_indices   
		          );
		 
		 
		 public BitonicSortAparapiKernel bsort_merge_last(//
		          Range _range,//
		          @GlobalWriteOnly(value = "g_data") float[] g_data,//
		          @Local(value = "l_data") float[] l_data,
		          @Constant(value="dir") int dir,
		          @GlobalWriteOnly( value ="g_indices") int[] g_indices,
		          @Local( value ="l_indices") int[] l_indices   
		          );
	}

	
	private int m_workgroup_size;
	private int m_local_size;
	
	private int[] m_indices_buffer;
	private float[] m_local_float_buffer;
	
	private BitonicSortAparapiKernel m_kernel;

	public BitonicSortAparapi(Device device)
	{
		m_kernel = ((OpenCLDevice)device).bind(BitonicSortAparapiKernel.class);
		m_workgroup_size = device.getMaxWorkGroupSize();
		m_local_size = (int)Math.pow(2, Math.floor(Math.log((float)m_workgroup_size)/Math.log(2.0f))); // align to power of 2
		m_local_float_buffer= new float[8 * m_local_size ];

	}
	
	public void sort(Device device, float[] array, int[] indices)
	{
		int direction = 0;
		m_indices_buffer = new int[array.length];

		//
		int local_size = m_local_size;
		int global_size = array.length /8; 
		if(global_size < local_size) {
			local_size   = global_size;
		}
		
		Range _range = device.createRange(global_size, local_size);
		m_kernel.bsort_init(_range, array, m_local_float_buffer, indices, m_indices_buffer);
		
		int num_stages = global_size/local_size;
		
		for(int high_stage = 2; high_stage < num_stages; high_stage <<= 1) {
			
		      for(int stage = high_stage; stage > 1; stage >>= 1) {
		    	  m_kernel.bsort_stage_n(_range, 
		    			  array, m_local_float_buffer, stage, high_stage, 
		    			  indices, m_indices_buffer);
		      }
		      
		      m_kernel.bsort_stage_0(_range, 
	    			  array, m_local_float_buffer, high_stage, 
	    			  indices, m_indices_buffer);
	   }
		

		   /* Perform the bitonic merge */
		   for(int stage =(int)num_stages; stage > 1; stage >>= 1) {
			   m_kernel.bsort_merge(_range, array, m_local_float_buffer, stage,
					   direction, indices, m_indices_buffer);
		   }
		   
		   int last_stage_size = (int)Math.max(local_size , indices.length / 8);
		   
		   Range _last_range = device.createRange(last_stage_size, local_size);
		   
		   m_kernel.bsort_merge_last(_last_range, array, m_local_float_buffer, direction, indices, m_indices_buffer);
		
	}
	
	
}
