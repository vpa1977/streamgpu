package org.stream_gpu.knn;

import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.opencl.CLMem.Usage;

public class PrefixDistance implements IDistance {
	
	private SlidingWindow m_window;
	private CLQueue m_calc_queue;
	CLBuffer<Float> m_output;
	private CLBuffer<Float> m_input;
	private CLBuffer<Float> m_ranges;
	private CLKernel m_distance_kernel;

	public PrefixDistance( SlidingWindow window , CLQueue queue, CLContext cl_context)
	{
		m_window = window;
		m_calc_queue = queue;
		m_output = cl_context.createBuffer(Usage.InputOutput,Float.class, m_window.getWindowSize());
		m_input = cl_context.createBuffer(Usage.Input, Float.class, m_window.getInstanceSize());
		m_ranges = cl_context.createBuffer(Usage.Input, Float.class, m_window.getRangesSize());
    	String source = null;
    	
    	try {
    		source = KernelLoader.readKernel("prefix_distance.cl");
    	} catch (Exception e){ e.printStackTrace();}
		m_distance_kernel = cl_context.createProgram(source).createKernel("prefix_distance");

	}

	@Override
	public CLBuffer<Float> distance(Instance inst, CLEvent event) {
		return null;
	}
	
	public static void main(String[] args)
	{
		CLContext context = JavaCL.createContext(null,
				JavaCL.listPlatforms()[0].listAllDevices(false)[0]);
		int window_size = 8192;
		int instance_size = 32;
		
		ArrayList<Attribute> attinfo = new ArrayList<Attribute>();
		for (int i = 0 ;i < instance_size ; i ++ )
		{
			attinfo.add( new Attribute(i+""));	
		}
		Instances dataset = new Instances( "test", attinfo , 1);
		dataset.setClassIndex(0);
		CLQueue queue = context.createDefaultQueue();
		SlidingWindow window = new SlidingWindow(context, window_size, dataset);
		for (int i = 0 ; i < window_size ; i++)
		{
			Instance newInstance = new DenseInstance(32);
			newInstance.setDataset(dataset);
			window.addInstance(queue, newInstance);
		}
		
	
	}

}
