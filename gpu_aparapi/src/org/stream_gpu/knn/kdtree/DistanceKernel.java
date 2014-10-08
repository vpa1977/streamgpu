package org.stream_gpu.knn.kdtree;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Range;
//import com.amd.aparapi.device.Device;
//import com.amd.aparapi.opencl.OpenCL.Write;
import com.amd.aparapi.device.Device;
import com.amd.aparapi.opencl.OpenCL.Write;
import com.amd.aparapi.opencl.OpenCL.Local;

public class DistanceKernel extends Kernel {
	
	public DistanceKernel(int instance_length, int numerics)
	{
		m_test = new float[instance_length];
		m_distances = new float[instance_length];
		m_instance_length = instance_length;
		m_numeric = numerics;
		setExplicit(true);
	}
	
	public void assign(float[] input_data, float[] test)
	{
		m_input_data = input_data;
		m_test = test;
		m_results = new float[m_input_data.length / m_instance_length] ;
		put(m_input_data);
		put(m_test);
	}
	
	public void compute(Device d, int k)
	{
		Range r = d.createRange2D(m_input_data.length / m_instance_length, m_instance_length);
		execute(r);
		get(m_results, 0 , k);
	}
	
	@Write
	public float[] m_input_data;
	@Write 
	public float[] m_test;
	public float[] m_results;
	public int m_instance_length;
	@Local
	private float[] m_distances;
	@Write
	public int m_numeric;

	@Override
	public void run() {
		float dist =0;
		int id = getGlobalId(1);
		
		dist = m_input_data[id] - m_test[id];
		
		if ( id >= m_numeric && dist > 0)
			dist = 1;
		
		m_distances[id] = dist*dist;
		// Step 2 - create intermediate sum by stepping in powers of 2 (1,2,4,8)
        for (int step = 2; step<=m_instance_length; step *= 2){
           if ((id+1)%step == 0){
              int stride = step/2;
              m_distances[id] += m_distances[id-stride];
           }
           localBarrier();
        }
        
        if (id == m_instance_length-1)
        	m_results[getGlobalId(0)] = m_distances[id];
	}

}
