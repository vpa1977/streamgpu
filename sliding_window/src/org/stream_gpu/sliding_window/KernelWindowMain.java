package org.stream_gpu.sliding_window;

import com.nativelibs4java.opencl.CLContext;

public class KernelWindowMain extends AbstractTest {

	@Override
	protected IFixedSlidingWindow makeWindow(CLContext context,
			int window_size, int instance_size) {
		return new KernelWindow(context, window_size, instance_size);
	}

	public static void main(String[] args) {
		new KernelWindowMain().test();
	}

}
