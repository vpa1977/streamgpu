package org.stream_gpu.knn.kdtree;

import java.util.ArrayList;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Range;
import com.amd.aparapi.device.Device;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class GpuInstances {
	
	private Instances m_dataset;
	private Integer[] m_numerics;
	private Integer[] m_nominals;
	private int m_instance_size;
	
	private Device m_device;
	private Range m_range;
	
	public GpuInstances(Instances model)
	{
		m_dataset = model;
		ArrayList<Integer> nominals = new  ArrayList<Integer>();
		ArrayList<Integer> numerics = new ArrayList<Integer>();
		int class_index = m_dataset.classIndex();
		for (int i = 0 ; i < m_dataset.numAttributes() ; i ++ )
		{
			if (i == class_index)
				continue;
			
			Attribute attr = m_dataset.attribute(i);
			if (attr.isNominal())
				nominals.add(i);
			if (attr.isNumeric())
				numerics.add(i);
		}
		m_numerics = numerics.toArray(new Integer[numerics.size()]);
		m_nominals = nominals.toArray(new Integer[nominals.size()]);
		m_instance_size = m_numerics.length + m_nominals.length;
		
		
		setDevice (Device.firstGPU());
	}
	
	public void setDevice(Device dev)
	{
		m_device = dev;
		m_range = m_device.createRange(m_numerics.length);
	}
	
	public GpuInstance createInstance(Instance inst)
	{
		float[] transfer = new float[ m_numerics.length + m_nominals.length];
		int offset = 0;
		for (int i = 0; i < m_numerics.length; i ++ ) 
		{
			int idx = m_numerics[i];
			transfer[offset++] = (float) inst.value(idx);
		}

		for (int i = 0; i < m_nominals.length; i ++ ) 
		{
			transfer[offset++] = (float) inst.value(m_nominals[i]);
		}
		return new GpuInstance(this,inst, transfer);
	}


	public int getNumericsLength() {
		return m_numerics.length;
	}
	
	public int length() {
		return m_numerics.length + m_nominals.length;
	}

	public Instances model() {
		return m_dataset;
	}
}
