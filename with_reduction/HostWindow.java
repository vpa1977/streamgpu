package org.stream_gpu.knn;

import java.util.ArrayList;

import org.bridj.Pointer;

import weka.core.Instance;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLMem.MapFlags;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLQueue;

/**
 * Maintains a window of fixed size in the GPU memory. Fastest implementation
 * maps by offset. Can probably be optimized by using correct alignment
 * 
 * @author ���
 * 
 */
public class HostWindow implements IFixedSlidingWindow {


	private CLBuffer<Float> m_data;
	private int m_index;
	private int m_window_size;
	private int m_instance_size;
	private Pointer<Float> m_mapped = null;
	private Instance[] m_instances;

	public HostWindow(CLContext context, int window_size, int instance_size) {
		m_index = -1;
		m_instance_size = (int)Math.pow(2, Math.ceil(Math.log((float)instance_size)/Math.log(2.0f))); // align to power of 2
		m_window_size = window_size;
		m_data = context.createFloatBuffer(Usage.Input, m_window_size* m_instance_size);
		m_instances = new Instance[window_size];

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.stream_gpu.sliding_window.IFixedSlidingWindow#addInstance(com.
	 * nativelibs4java.opencl.CLQueue, float[])
	 */
	@Override
	public CLEvent addInstance(CLQueue queue, Instance inst, float[] instance) throws Exception {
		m_index++;
		if (m_index >= m_window_size)
			m_index = 0;
		if (instance.length == inst.toDoubleArray().length)
			throw new Exception("Invalid attribute array");
		m_instances[m_index] = inst;
		m_mapped = m_data.map(queue, MapFlags.Write, m_index * m_instance_size,
				m_instance_size, new CLEvent[0]);
		m_mapped.setFloats(instance);
		return m_data.unmap(queue, m_mapped, new CLEvent[0]);

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * org.stream_gpu.sliding_window.IFixedSlidingWindow#close(com.nativelibs4java
	 * .opencl.CLQueue)
	 */
	@Override
	public void close(CLQueue queue) {
		m_data.release();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.stream_gpu.sliding_window.IFixedSlidingWindow#getWindowSize()
	 */
	@Override
	public int getWindowSize() {
		return m_window_size;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.stream_gpu.sliding_window.IFixedSlidingWindow#getInstanceSize()
	 */
	@Override
	public int getInstanceSize() {
		return m_instance_size;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.stream_gpu.sliding_window.IFixedSlidingWindow#getIndex()
	 */
	@Override
	public int getIndex() {
		return m_index;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.stream_gpu.sliding_window.IFixedSlidingWindow#getBuffer()
	 */
	@Override
	public CLBuffer<Float> getBuffer() {
		return m_data;
	}
	

	public void resetInstances() {
		m_instances = new Instance[ m_window_size];
	}
	
	public Instance[] instances() 
	{
		return m_instances;
	
	}

}
