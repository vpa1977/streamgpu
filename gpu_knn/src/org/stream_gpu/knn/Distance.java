package org.stream_gpu.knn;

import java.io.BufferedReader;
import java.io.InputStreamReader;

import weka.core.Instance;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.CLMem.Usage;

public class Distance implements IDistance {
	
	private SlidingWindow m_window;
	private CLQueue m_calc_queue;
	CLBuffer<Float> m_output;
	private CLBuffer<Float> m_input;
	private CLBuffer<Float> m_ranges;
	private CLKernel m_distance_kernel;

	public Distance( SlidingWindow window , CLQueue queue, CLContext cl_context)
	{
		m_window = window;
		m_calc_queue = queue;
		m_output = cl_context.createBuffer(Usage.InputOutput,Float.class, m_window.getWindowSize());
		m_input = cl_context.createBuffer(Usage.Input, Float.class, m_window.getInstanceSize());
		m_ranges = cl_context.createBuffer(Usage.Input, Float.class, m_window.getRangesSize());
    	String source = null;
    	
    	try {
    		source = readKernel("distance.cl");
    	} catch (Exception e){ e.printStackTrace();}
		m_distance_kernel = cl_context.createProgram(source).createKernel("square_distance");

	}
	
	 /* (non-Javadoc)
	 * @see org.stream_gpu.knn.IDistance#distance(weka.core.Instance, com.nativelibs4java.opencl.CLEvent)
	 */
	@Override
	public CLBuffer<Float> distance(Instance inst, CLEvent event) 
	    {
	    	if (m_window == null)
	    		throw new IllegalArgumentException();
	    	
	    	CLEvent input_ready = m_window.storeTargetInstance(m_calc_queue, inst,  m_input, m_ranges);
	    	
			m_distance_kernel.setArgs(
							 m_input,
							 m_window.getBuffer(),
							 m_ranges,
							 m_output, 
							 m_window.getInstanceSize(), 
							 m_window.numericsSize(), 
							 m_window.nominalsSize());

	    	CLEvent distance_done = m_distance_kernel.enqueueNDRange(m_calc_queue,
	    			 			null, // offsets
	    						new long[] { m_window.getWindowSize() }, // global sizes 
	    						null, // local sizes
	    			 new CLEvent[]{ event , input_ready} ); 
			m_calc_queue.finish();
			return m_output;
	    }
	 
		private static String readKernel(String name) throws Exception {
			BufferedReader r = new BufferedReader(new InputStreamReader(KnnGpuClassifier.class.getResourceAsStream(name)));
			String output = "";
			String line;
			while ((line = r.readLine()) != null)
				output += line + "\n";
			r.close();
			return output;

		}

}
