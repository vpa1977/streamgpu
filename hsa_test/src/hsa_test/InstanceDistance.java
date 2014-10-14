package hsa_test;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;












import org.bridj.Pointer;

import com.amd.aparapi.Aparapi;
import com.amd.aparapi.Aparapi.IntTerminal;
import com.amd.aparapi.Device;
import com.amd.okra.OkraContext;
import com.amd.okra.OkraKernel;
import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem.MapFlags;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;

public class InstanceDistance {
	
	
	public float[] simple_distance(final float[][] nodes, float[] v2, final int numerics, final int instance_len )
	{
		float[] res = new float[ nodes.length];
		for (int i = 0 ;i < nodes.length ; ++i)
		{
			res[i] = 0;
			for (int j = 0 ; j < instance_len; ++j)
			{
				float val = (nodes[i][j] - v2[j]); 
				res[i] += val*val;
			}
		}
		return res;
	}
	
	public float[] distance(Device dev, final float[][] nodes, float[] v2, final int numerics, final int instance_len)
	{
		final int len = nodes.length * instance_len;
		final float[][] distance_vector = new float[nodes.length][instance_len];
		final float[] result = new float[nodes.length];
		final IntTerminal ic = id -> {
				int instance = id / instance_len;
				int local_id = id % instance_len;
				float[] sample = nodes[instance];
				distance_vector[instance][local_id] = (sample[local_id] - v2[local_id]);
				distance_vector[instance][local_id]*= distance_vector[instance][local_id];
				int step =1;
				do {
					step *= 2;
			         if ((local_id+1)%step == 0){
				         int stride = step/2;
				          distance_vector[instance][local_id] += distance_vector[instance][local_id-stride];
				     }
					
				} while (step <= instance_len);
				if (local_id == instance_len -1)
					result[ id / instance_len] = distance_vector[instance][local_id];
			};
		dev.forEach(0,len, ic);
		return result;
	}
	

	
	/**
	 * Compute Distance using Lambda binding 
	 * @param dev
	 * @param nodes
	 * @param v2
	 * @param numerics
	 * @param instance_len
	 * @return
	 */
	public float[] distance_hsa(Device dev, final float[][] nodes, float[] v2, final int numerics, final int instance_len)
	{
		final float[] result = new float[nodes.length];
		final IntTerminal ic = id -> {
			float res = 0;
			int i = 0;
			do {
				float val = nodes[id][i] - v2[i];
				res += val*val;
				++i;
			} while (i < instance_len);
			result[id] = res;
		};
		dev.forEach(0,result.length, ic);
		return result;
	}
	
	/** 
	 * Compute distance using okra binding
	 * @return
	 */
	public float[] distance_okra(final float[] samples, final float[] vector)
	{
		final float[] result = new float[samples.length/ vector.length];
		kernel.clearArgs();
		kernel.pushIntArg(0);
		kernel.pushIntArg(0);
		kernel.pushIntArg(0);
		kernel.pushIntArg(0);
		kernel.pushIntArg(0);
		kernel.pushIntArg(0);
		
		kernel.pushFloatArrayArg(samples);
		kernel.pushFloatArrayArg(vector);
		kernel.pushFloatArrayArg(result);
		kernel.pushIntArg(16);
		kernel.setLaunchAttributes(samples.length/ vector.length, 512);
		kernel.dispatchKernelWaitComplete();
		return result;
	}
	
	public void loadOcl() throws Throwable
	{
		BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream("distance.cl")));
		String output = "";
		String line;
		while ((line = r.readLine()) != null)
			output += line + "\n";
		r.close();
		
		
		CLDevice[] device = JavaCL.listPlatforms()[0].listAllDevices(false);
		System.out.println(device[0].getName());
		m_context = JavaCL.createContext(null,	device[0]);
		m_queue = m_context.createDefaultQueue();
		CLProgram program = m_context.createProgram(output);
		m_cl_kernel = program.createKernel("square_distance");

		samples_buffer = m_context.createFloatBuffer(Usage.Input, 128*1024);
		vector_buffer = m_context.createFloatBuffer(Usage.Input, 16);
		result_buffer = m_context.createFloatBuffer(Usage.Output, 128*1024/16);
 
	}
	
	public float[] distance_ocl(final float[] samples, final float[] vector)
	{
		final float[] result = new float[samples.length/ vector.length];
		
		Pointer<Float> ptr = samples_buffer.map(m_queue, MapFlags.Write,0, samples.length, null);
		ptr.setFloats(samples);
		samples_buffer.unmap(m_queue,ptr,null);
		
		ptr = vector_buffer.map(m_queue, MapFlags.Write, 0, vector.length, null);
		ptr.setFloats(samples);
		vector_buffer.unmap(m_queue,ptr,null);

		m_queue.flush();
		m_cl_kernel.setArg(0, samples_buffer);
		m_cl_kernel.setArg(1, vector_buffer);
		m_cl_kernel.setArg(2, result_buffer);
		m_cl_kernel.setArg(3, vector.length );
		
		m_cl_kernel.enqueueNDRange(m_queue, new int[] { result.length }, null);
		m_queue.flush();
		Pointer<Float> res = result_buffer.map(m_queue, MapFlags.Read, 0,samples.length/ vector.length, null);
		res.getFloats(result);
		result_buffer.unmap(m_queue, res, null);
		return result;
	}
	
	public void loadOkra() throws Throwable
	{
		BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream("distance.hsail")));
		String output = "";
		String line;
		while ((line = r.readLine()) != null)
			output += line + "\n";
		r.close();
		kernel = new OkraKernel(context, output,"&__OpenCL_square_distance_kernel" );
		
	}
	
	private OkraContext context = new OkraContext();
	private OkraKernel kernel;
	private CLContext m_context;
	private CLQueue m_queue;
	private CLKernel m_cl_kernel;
	private CLBuffer<Float> samples_buffer;
	private CLBuffer<Float> vector_buffer;
	private CLBuffer<Float> result_buffer;
}
