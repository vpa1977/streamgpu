package org.stream_gpu.knn;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.Map;

import org.bridj.Pointer;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.LocalSize;
import com.nativelibs4java.opencl.CLMem.Flags;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;

public class BitonicSort {
	
	private long m_workgroup_size;
	private long m_local_size;
	
	private CLContext m_context;
	private CLQueue m_sort_queue;
	private CLKernel m_sort_init;
	private CLKernel m_sort_stage_0;
	private CLKernel m_sort_stage_n;
	private CLKernel m_sort_merge;
	private CLKernel m_sort_merge_last;
	private CLBuffer<Float> m_io_buffer;
	private CLBuffer<Integer> m_indices_buffer;

	public BitonicSort(CLContext cl_context, int length)
	{
		m_context  = cl_context;
		String source = null;
    	try {
    		source = readKernel("sort.cl");
    	} catch (Exception e){ e.printStackTrace();}
		CLProgram program  = m_context.createProgram(source);
		// create sort kernels
		m_sort_init = program.createKernel("bsort_init");
		m_sort_stage_0 = program.createKernel("bsort_stage_0");
		m_sort_stage_n = program.createKernel("bsort_stage_n");
		m_sort_merge = program.createKernel("bsort_merge");
		m_sort_merge_last = program.createKernel("bsort_merge_last");
		m_workgroup_size = max_workgroup_size();
		m_local_size = (int)Math.pow(2, Math.floor(Math.log((float)m_workgroup_size)/Math.log(2.0f))); // align to power of 2
		m_sort_queue = m_context.createDefaultQueue();
		m_indices_buffer = m_context.createBuffer(Usage.InputOutput,Integer.class, length);
		m_io_buffer = m_context.createBuffer(Usage.InputOutput,Float.class, length);
		
		
		// init io buffer arg
		m_sort_init.setArg(0, m_io_buffer);
		m_sort_init.setArg(2, m_indices_buffer);
		m_sort_init.setArg(3, LocalSize.ofIntArray(8*m_local_size));
		
		m_sort_stage_0.setArg(0, m_io_buffer);
		
		m_sort_stage_0.setArg(3, m_indices_buffer);
		m_sort_stage_0.setArg(4, LocalSize.ofIntArray(8*m_local_size));
		
		m_sort_stage_n.setArg(0, m_io_buffer);
		
		m_sort_stage_n.setArg(4, m_indices_buffer);
		m_sort_stage_n.setArg(5, LocalSize.ofIntArray(8*m_local_size));
		
		
		m_sort_merge.setArg(0, m_io_buffer);
		
		m_sort_merge.setArg(4, m_indices_buffer);
		m_sort_merge.setArg(5, LocalSize.ofIntArray(8*m_local_size));
		
		m_sort_merge_last.setArg(0, m_io_buffer);
		
		m_sort_merge_last.setArg(3, m_indices_buffer);
		m_sort_merge_last.setArg(4, LocalSize.ofIntArray(8*m_local_size));

		//
		m_sort_init.setArg(1, LocalSize.ofFloatArray(8 * m_local_size));
		m_sort_stage_0.setArg(1, LocalSize.ofFloatArray(8 * m_local_size));
		m_sort_stage_n.setArg(1, LocalSize.ofFloatArray(8 * m_local_size));
		m_sort_merge.setArg(1, LocalSize.ofFloatArray(8 * m_local_size));
		m_sort_merge_last.setArg(1, LocalSize.ofFloatArray(8 * m_local_size));
		//
		
	}
	
	public float[] sort(float[] array, int[] indices)
	{
		
		int direction = 0;
	    
		// create buffer and map it
	    
	    Pointer<Integer> p_indices = m_indices_buffer.map(m_sort_queue, CLMem.MapFlags.Write, new CLEvent[0]);
	    p_indices.setInts(indices);
	    m_indices_buffer.unmap(m_sort_queue, p_indices, new CLEvent[0]);
	    
		
		Pointer<Float> p_float = m_io_buffer.map(m_sort_queue, CLMem.MapFlags.Write, new CLEvent[0]);
		p_float.setFloats(array);
		m_io_buffer.unmap(m_sort_queue, p_float, new CLEvent[0]);
		
		long local_size = m_local_size;
		long global_size = array.length /8; 
		if(global_size < local_size) {
			local_size   = global_size;
		}
		m_sort_init.enqueueNDRange(m_sort_queue, new long[]{} ,  new long[]{ global_size} , new long[] { local_size } , new CLEvent[0]);

		long num_stages = global_size/local_size;
		
		 for(int high_stage = 2; high_stage < num_stages; high_stage <<= 1) {
			 
			 m_sort_stage_0.setArg(2, high_stage);
			 m_sort_stage_n.setArg(3, high_stage);


		      for(int stage = high_stage; stage > 1; stage >>= 1) {
		    	  m_sort_stage_n.setArg(2, stage);
		    	  m_sort_stage_n.enqueueNDRange(m_sort_queue, 
		    			  new long[]{},
		    			  new long[]{ global_size}, 
		    			  new long[]{ local_size }, 
		    			  new CLEvent[0]);
		      }
		      
		      m_sort_stage_0.enqueueNDRange(m_sort_queue,
	    			  new long[]{},
	    			  new long[]{ global_size}, 
	    			  new long[]{ local_size }, 
	    			  new CLEvent[0]);
		   }

		    
		   /* Set the sort direction */
       	m_sort_merge.setArg(3, direction);
		m_sort_merge_last.setArg(2, direction); 	

		   /* Perform the bitonic merge */
		   for(long stage = num_stages; stage > 1; stage >>= 1) {
			   m_sort_merge.setArg(2, stage);
			   m_sort_merge.enqueueNDRange(m_sort_queue, 
					   new long[] {},
					   new long[] { global_size }, 
					   new long[] { local_size }, 
					   new CLEvent[0]);
		   }
		   
		   m_sort_merge_last.enqueueNDRange(m_sort_queue, 
				   new long[] {},
				   new long[] { global_size }, 
				   new long[] { local_size }, 
				   new CLEvent[0]);

		   
		  m_sort_queue.finish();

			p_float = m_io_buffer.map(m_sort_queue, CLMem.MapFlags.Read, new CLEvent[0]);
			p_float.getFloats(array);
			m_io_buffer.unmap(m_sort_queue, p_float, new CLEvent[0]);
			//m_io_buffer.release();
			
			p_indices =  m_indices_buffer.map(m_sort_queue,  CLMem.MapFlags.Read, new CLEvent[0]);
			p_indices.getInts(indices);
			m_indices_buffer.unmap(m_sort_queue, p_indices, new CLEvent[0]);
			return array;
	}

	private long max_workgroup_size() {
		Map<CLDevice, Long> sizes = m_sort_init.getWorkGroupSize();
		long workgroup_size = Long.MAX_VALUE;
		for (Long l : sizes.values())
		{
			if (l < workgroup_size )
			{
				workgroup_size  = l;
			}
		}
		return workgroup_size;
	}
	
	public static String readKernel(String name) throws Exception {
		BufferedReader r = new BufferedReader(new InputStreamReader(KnnGpuClassifier.class.getResourceAsStream(name)));
		String output = "";
		String line;
		while ((line = r.readLine()) != null)
			output += line + "\n";
		r.close();
		return output;

	}

	
	
}
