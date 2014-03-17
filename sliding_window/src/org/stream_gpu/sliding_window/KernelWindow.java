package org.stream_gpu.sliding_window;

import org.bridj.Pointer;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem.MapFlags;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLQueue;

/**
 * Kernel-based implementation of fixed size sliding window. (+500 ms to copy
 * operaiton) If we need to maintain histogram, probably this one will be
 * cheapest.
 * 
 * @author ÐÐÐ
 * 
 */
public class KernelWindow implements IFixedSlidingWindow {

	private CLBuffer<Float> m_data;
	private CLBuffer<Float> m_instance;
	private int m_index;
	private int m_window_size;
	private int m_instance_size;
	private CLKernel m_kernel;

	public KernelWindow(CLContext context, int window_size, int instance_size) {
		m_index = -1;
		m_instance_size = instance_size;
		m_window_size = window_size;
		m_data = context.createFloatBuffer(Usage.InputOutput, window_size
				* instance_size);
		m_instance = context.createFloatBuffer(Usage.Input, instance_size);

		String source = "__kernel void move_window(__global const float* input,  __global float* output) \n"
				+ "{\n"
				+ "int offset = get_global_id(0);\n "
				+ "int gl_offset = get_global_id(1);\n"
				+ " output[offset + gl_offset] = input[offset];\n" + "}\n";
		m_kernel = context.createProgram(source).createKernel("move_window");
	}

	/**
	 * Currently addInstance uses main buffer. Should we use large window/large
	 * instances we should use createSubBufer
	 * 
	 * @param queue
	 * @param instance
	 */
	public void addInstance(CLQueue queue, float[] instance) {
		m_index++;
		if (m_index >= m_window_size)
			m_index = 0;
		Pointer<Float> floatPointer = m_instance.map(queue, MapFlags.Write,
				new CLEvent[0]);
		floatPointer.setFloats(instance);
		m_instance.unmap(queue, floatPointer, new CLEvent[0]);
		m_kernel.setArgs(m_instance, m_data);
		m_kernel.enqueueNDRange(queue, new long[] { 0,
				m_instance_size * m_index }, new long[] { m_instance_size, 1 },
				null, new CLEvent[0]);
	}

	/**
	 * perform clean up if needed
	 * 
	 * @param queue
	 */
	public void close(CLQueue queue) {
	}

	public int getWindowSize() {
		return m_window_size;
	}

	public int getInstanceSize() {
		return m_instance_size;
	}

	public int getIndex() {
		return m_index;
	}

	public CLBuffer<Float> getBuffer() {
		return m_data;
	}
}
