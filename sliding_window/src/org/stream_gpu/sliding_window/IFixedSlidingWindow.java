package org.stream_gpu.sliding_window;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLQueue;

public interface IFixedSlidingWindow {

	/**
	 * Currently addInstance uses main buffer. Should we use large window/large
	 * instances we should use createSubBufer
	 * 
	 * @param queue
	 * @param instance
	 */
	public abstract void addInstance(CLQueue queue, float[] instance);

	/**
	 * perform clean up if needed
	 * 
	 * @param queue
	 */
	public abstract void close(CLQueue queue);

	public abstract int getWindowSize();

	public abstract int getInstanceSize();

	public abstract int getIndex();

	public abstract CLBuffer<Float> getBuffer();

}