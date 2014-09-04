package org.stream_gpu.knn.kdtree;

import weka.core.Instance;

/** 
 * Convert weka instance into float representation for GPU
 * @author ÐÐÐ
 *
 */
public class GpuInstance {
	private Instance m_instance;
	private GpuInstances m_model;
	private float[] m_values;
	private double m_distance;

	public GpuInstance(GpuInstances model, Instance instance, float[] values)
	{
		m_instance = instance;
		m_model = model;
		m_values = values;
	}

	public float[] data() {
		return m_values;
	}

	public void setDistance(float d) {
		m_distance = d;
	}
	
	public double distance() {
		return m_distance;
	}

	public Instance wekaInstance() {
		return m_instance;
	}
}
