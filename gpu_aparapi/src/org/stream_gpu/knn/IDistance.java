package org.stream_gpu.knn;

import weka.core.Instance;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLEvent;

public interface IDistance {

	public abstract CLBuffer<Float> distance(Instance inst, CLEvent event);

}