package org.stream_gpu.sliding_window;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import org.bridj.Pointer;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem.MapFlags;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;

public abstract class AbstractTest {

	public static String readKernel() throws Exception {
		BufferedReader r = new BufferedReader(new FileReader("simple.cl"));
		String output = "";
		String line;
		while ((line = r.readLine()) != null)
			output += line + "\n";
		r.close();
		return output;

	}

	protected abstract IFixedSlidingWindow makeWindow(CLContext context,
			int window_size, int instance_size);

	protected void test() {
		int instance_size = 8192;
		int window_size = 1024;
		try {

			float[] array = new float[instance_size];
			for (int i = 0; i < instance_size; i++)
				array[i] = 1;

			float[] array1 = new float[instance_size];
			for (int i = 0; i < instance_size; i++)
				array1[i] = 0;

			CLContext context = JavaCL.createContext(null,
					JavaCL.listPlatforms()[0].listAllDevices(false)[0]);
			CLQueue queue = context.createDefaultQueue();
			IFixedSlidingWindow theWindow = makeWindow(context, window_size,
					instance_size);
			String source = readKernel();
			CLKernel kernel = context.createProgram(source).createKernel(
					"addFloats");
			CLBuffer<Float> output = context.createBuffer(Usage.Output,
					Float.class, theWindow.getInstanceSize());

			for (int i = 0; i < window_size; i++)
				theWindow.addInstance(queue, array);

			// computing 1000 times
			long start = System.currentTimeMillis();
			for (int i = 0; i < 1000; i++) {
				theWindow.addInstance(queue, array);
				theWindow.addInstance(queue, array1);
			//	kernel.setArgs(theWindow.getBuffer(),
			//			theWindow.getInstanceSize(), theWindow.getWindowSize(),
			//			output);
			//	kernel.enqueueNDRange(
			//			queue,
			//			new long[] { 0, 0 },
			//			new long[] { theWindow.getInstanceSize(),
			//					theWindow.getWindowSize() }, null,
			//			new CLEvent[0]);
			}
			queue.finish();
			Pointer<Float> result = output.map(queue, MapFlags.Read,
					new CLEvent[0]);
			float[] result_array = result.getFloats();
			output.unmap(queue, result, new CLEvent[0]);
			theWindow.close(queue);
			long end = System.currentTimeMillis();

			for (int i = 0; i < result_array.length; i++)
				System.out.print(result_array[i] + ",");
			System.out.println();
			System.out.println("Elapsed : " + (end - start));
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	private static Random m_random = new Random();

	/**
	 * creates a random double array with defined length
	 * 
	 * @return
	 */
	public static double[] makeInstance(int len) {
		double[] ret = new double[len];
		for (int i = 0; i < ret.length; i++)
			ret[i] = m_random.nextDouble();
		return ret;
	}

}
