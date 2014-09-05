package org.stream_gpu.knn.kdtree;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Range;
import com.amd.aparapi.device.Device;
import com.amd.aparapi.opencl.OpenCL.Write;

public class DistanceKernel extends Kernel {
	
	public DistanceKernel(int instance_length)
	{
		m_test = new float[instance_length];
		m_distances = new float[instance_length];
		m_instance_length = instance_length;
		setExplicit(true);
	}
	
	public void assign(float[] input_data)
	{
		m_input_data = input_data;
		m_results = new float[m_input_data.length / m_instance_length] ;
		put(m_input_data, 0, m_input_data.length);
	}
	
	public void compute(Device d)
	{
		Range r = d.createRange2D(m_instance_length, m_input_data.length / m_instance_length);
		execute(r);
		get(m_results);
	}
	
	@Write
	public float[] m_input_data;
	
	@Write 
	public float[] m_test;
	public float[] m_results;
	public int m_instance_length;
	
	public float[] m_distances;


	@Override
	public void run() {
		int id = getGlobalId(0);
		float dist = m_input_data[id] - m_test[id];
		m_distances[id] = dist*dist;
		// Step 2 - create intermediate sum by stepping in powers of 2 (1,2,4,8)
        for (int step = 2; step<=m_instance_length; step *= 2){
           if ((id+1)%step == 0){
              int stride = step/2;
              m_distances[id] += m_distances[id-stride];
           }
        }
        if (id == m_instance_length-1)
        	m_results[getGlobalId(1)] = m_distances[id];
	}

}
