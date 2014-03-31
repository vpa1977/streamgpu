package org.stream_gpu.knn;

import weka.core.Instance;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLQueue;

public interface IFixedSlidingWindow {

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

	public abstract CLEvent addInstance(CLQueue queue, Instance inst, float[] instance);

}