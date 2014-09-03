package org.stream_gpu.knn.kdtree;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.opencl.OpenCL.Write;

import weka.core.EuclideanDistance;
import weka.core.Instances;

public class GpuDistance extends EuclideanDistance {
	
	private RangeUpdateKernel m_range_update;
	private GpuInstances m_model;

	public GpuDistance(GpuInstances dataset) {
		super(dataset.model());
		m_model = dataset;
		m_range_update = new RangeUpdateKernel(m_model);
		
	}
	
	
	private class RangeUpdateKernel extends Kernel 
	{
		
		public RangeUpdateKernel(GpuInstances model)
		{
			setExplicit(true);
			m_ranges = new float[model.getNumericsLength()*2];
			m_instance_data  = new float[model.getNumericsLength()];
			
			for (int i = 0; i < m_ranges.length/2 ; i ++)
			{
				int min = i *2;
				int max = min+1;
				m_ranges[min] = Float.MAX_VALUE;
				m_ranges[max] = Float.MIN_VALUE;
			}
			put(m_ranges);
		}
		
		public void put(GpuInstance gpuInst) {
			float[] data = gpuInst.data();
			System.arraycopy(data, 0, m_instance_data, 0, m_instance_data.length);
			put(m_instance_data);
		}

		@Write
		private float[] m_ranges;
		@Write
		private float[] m_instance_data;

		@Override
		public void run() {
			int field = getGlobalId();
			int min = field *2;
			int max = min+1;
			float value = m_instance_data[field];
			if (value > m_ranges[max])
				m_ranges[max] = value;
			if (value < m_ranges[min])
				m_ranges[min] = value;
		}
	}

}
