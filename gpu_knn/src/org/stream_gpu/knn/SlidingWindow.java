package org.stream_gpu.knn;

import java.util.ArrayList;

import org.bridj.Pointer;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.LocalSize;
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
public class SlidingWindow  {


	private static final int FLOAT_SIZE = 4;
	private CLBuffer<Float> m_data;
	private int m_index;
	private boolean m_ready;
	private int m_flush_index;
	private int m_window_size;
	private int m_instance_size;
	private Pointer<Float> m_mapped = null;
	private Instance[] m_instances;
	private Instances m_dataset;
	private Integer[] m_numerics;
	private Integer[] m_nominals;
	private float[] m_ranges;
	private boolean m_full_flush;
	private int m_count;

	public SlidingWindow(CLContext context, int window_size, Instances dataset) {
		m_index = -1;
		m_window_size = window_size;
		m_dataset = dataset;
		m_flush_index = 0;
		m_count = 0;
		m_ready = false;
		m_full_flush = false;
		prepare(context);
	}
	
	
	private void prepare(CLContext context) 
	{
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
		m_ranges = new float[ m_numerics.length * 2];
		for (int i = 0 ;i < m_ranges.length ; i +=2 )
		{
			m_ranges[i] = Float.MAX_VALUE;
			m_ranges[i+1] = Float.MIN_VALUE;
		}
		m_instance_size = numericsSize() + nominalsSize();
		
		
		m_data = context.createFloatBuffer(Usage.Input, m_window_size* m_instance_size);
		m_instances = new Instance[m_window_size];
	}
	
	public int numericsSize() 
	{
		return m_numerics.length;
	}
	
	public int nominalsSize() 
	{
		return m_nominals.length;
	}
	
	public CLEvent flushInstances(CLQueue queue)
	{
		CLEvent ret = null;
		if (m_flush_index == m_index)
			return ret;
		
		if (m_full_flush)
		{
			ret = flush(queue,0, m_window_size-1);
		}
		else if (m_index < m_flush_index) 
		{
			ret = flush(queue,0, m_index);
			ret = flush(queue,m_flush_index, m_window_size-1);
		} 
		else 
		{
			ret = flush(queue,m_flush_index , m_index);
		}
		m_full_flush = false;
		m_flush_index = m_index;
		return ret;
	}

	private CLEvent flush(CLQueue queue, int start_index, int end_index) {
		m_mapped = m_data.map(queue, MapFlags.Write, m_instance_size * start_index, (1+end_index -start_index ) *	m_instance_size, new CLEvent[0]);
		for (int i = start_index ; i <=end_index; i ++ )
		{
			float[] transfer = makeTransferArray(m_instances[i]);
			updateRanges(transfer, m_ranges);
			m_mapped.setFloatsAtOffset((i-start_index) * transfer.length * FLOAT_SIZE,transfer );	
		}
		return m_data.unmap(queue, m_mapped, new CLEvent[0]);
	}


	/*
	 * (non-Javadoc)
	 * 
	 * @see org.stream_gpu.sliding_window.IFixedSlidingWindow#addInstance(com.
	 * nativelibs4java.opencl.CLQueue, float[])
	 */
	
	public void addInstance(CLQueue queue, Instance inst) {
		m_index++;
		if (m_index >= m_window_size) {
			m_index = 0;
			m_ready = true;
		}
		
		if (m_index == m_flush_index && m_instances[0]!=null) 
			m_full_flush = true;
		
		m_instances[m_index] = inst;
		
		tryFlush(queue);
	}
	
	public void tryFlush(CLQueue queue) 
	{
		m_count ++;
		if (m_count * getInstanceSize() * 4 >= 512) {
			flushInstances(queue);
			m_count = 0;
		}
		
	}


	private float[] makeTransferArray(Instance inst) {
		float[] transfer = new float[ numericsSize() + nominalsSize()];
		int offset = 0;
		for (int i = 0; i < m_numerics.length; i ++ ) 
		{
			int idx = m_numerics[i];
			if (inst == null){
				System.out.println("WHat !!~!");
			}
			transfer[offset++] = (float) inst.value(idx);
		}

		for (int i = 0; i < m_nominals.length; i ++ ) 
		{
			transfer[offset++] = (float) inst.value(m_nominals[i]);
		}
		return transfer;
	}
	
	public CLEvent storeTargetInstance(CLQueue queue, Instance inst, CLBuffer<Float> instance, 
						CLBuffer<Float> range_buffer) {
		float[] transfer = makeTransferArray(inst);
		m_mapped = instance.map(queue, MapFlags.Write, new CLEvent[0]);
		m_mapped.setFloats(transfer);
		instance.unmap(queue, m_mapped, new CLEvent[0]);
		m_mapped = range_buffer.map(queue, MapFlags.Write, new CLEvent[0]);
		float[] ranges = new float[m_ranges.length];
		System.arraycopy(m_ranges, 0, ranges, 0, ranges.length);
		updateRanges(transfer, ranges);
		m_mapped.setFloats(ranges);
		return range_buffer.unmap(queue, m_mapped, new CLEvent[0]);
	}
	

	private void updateRanges(float[] transfer, float[] ranges) {
	 for (int i = 0 ;i < numericsSize() ; i ++ ) 
	 {
		 float value = transfer[i];
		 if (Float.isNaN(value) || Float.isInfinite(value))
			 continue;
		 int min_i = i*2;
		 if (ranges[min_i] > value )
			 ranges[min_i] = value;
		 if (ranges[min_i+1] < value)
			 ranges[min_i+1] = value;
	 }
	}


	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * org.stream_gpu.sliding_window.IFixedSlidingWindow#close(com.nativelibs4java
	 * .opencl.CLQueue)
	 */
	
	public void close(CLQueue queue) {
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.stream_gpu.sliding_window.IFixedSlidingWindow#getWindowSize()
	 */
	
	public int getWindowSize() {
		return m_window_size;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.stream_gpu.sliding_window.IFixedSlidingWindow#getInstanceSize()
	 */
	
	public int getInstanceSize() {
		return m_instance_size;
	}

	public int getRangesSize() {
		return getInstanceSize()*2;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.stream_gpu.sliding_window.IFixedSlidingWindow#getIndex()
	 */
	
	public int getIndex() {
		return m_index;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.stream_gpu.sliding_window.IFixedSlidingWindow#getBuffer()
	 */
	
	public CLBuffer<Float> getBuffer() {
		return m_data;
	}

	public void resetInstances() {
		
	}
	
	public Instance[] instances() 
	{
		return m_instances;
	
	}


	public boolean ready() {
		return m_ready;
	}



}
