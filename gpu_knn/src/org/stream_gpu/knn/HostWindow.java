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
 * @author ÐÐÐ
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
		m_instance_size = instance_size;
		m_window_size = window_size;
		m_data = context.createFloatBuffer(Usage.Input, window_size
				* instance_size);
		m_instances = new Instance[window_size];

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.stream_gpu.sliding_window.IFixedSlidingWindow#addInstance(com.
	 * nativelibs4java.opencl.CLQueue, float[])
	 */
	@Override
	public CLEvent addInstance(CLQueue queue, Instance inst, float[] instance) {
		m_index++;
		if (m_index >= m_window_size)
			m_index = 0;
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
		
	}
	
	public Instance[] instances() 
	{
		return m_instances;
	
	}

}
