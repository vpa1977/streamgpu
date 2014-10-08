package org.stream_gpu.knn.kdtree;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Range;
import com.amd.aparapi.device.Device;
//import com.amd.aparapi.device.Device;
import com.amd.aparapi.opencl.OpenCL.Write;

public class DistanceKernel2 extends Kernel{
	
	public DistanceKernel2(int instance_length)
	{
		m_test = new float[instance_length];
		m_instance_length = instance_length;
		setExplicit(true);
	}
	
	public void assign(float[] input_data, float[] test)
	{
		m_input_data = input_data;
		m_test = test;
		//if (m_results == null)
		put(m_input_data);
		m_results = new float[m_input_data.length / m_instance_length] ;
	}
	
	public void compute(Device d, int k)
	{
		Range r = d.createRange( m_results.length );
		execute(r);
		get(m_results,0,k);
	}
	
	@Write
	public float[] m_input_data;
	
	@Write 
	public float[] m_test;
	public float[] m_results;
	public int m_instance_length;


	@Override
	public void run() {
		int id = getGlobalId(0);
		int vector_offset = id * m_instance_length;
		float dist = 0;
		float result = 0;
		for (int i = 0 ; i < m_instance_length; ++i)
		{
			int offset =  vector_offset + i;
			dist = m_input_data[offset] - m_test[i];
			result += dist * dist;
		}
       	m_results[id] =result;
       	
	}
}
