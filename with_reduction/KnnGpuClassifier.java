package org.stream_gpu.knn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Map;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;

import org.bridj.Pointer;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem.MapFlags;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.LocalSize;

public class KnnGpuClassifier extends AbstractClassifier {

	private static final long serialVersionUID = 1L;
	/** no weighting. */
	public static final int WEIGHT_NONE = 1;
	/** weight by 1/distance. */
	public static final int WEIGHT_INVERSE = 2;
	/** weight by 1-distance. */
	public static final int WEIGHT_SIMILARITY = 4;

	private long m_workgroup_size;
	private long m_local_size;

	private HostWindow m_window;
	private int m_k;
	private CLQueue m_data_transfer_queue;
	private CLQueue m_calc_queue;
	private CLContext m_cl_context;
	private CLEvent m_last_event;
	private CLKernel m_distance_kernel;
	private CLKernel m_reduction_kernel;
	private CLBuffer<Float> m_output;
	private CLBuffer<Float> m_input;
	private CLBuffer<Float> m_dist_vector;
	private int m_window_size;
	private int m_num_classes;
	private int m_num_attributes;
	private int m_class_type;
	private int[] m_indices;
	private BitonicSort m_sorter;
	private int m_num_attributes_used;
	private int m_distance_weighting;

	@Override
	public String getPurposeString() {
		return "kNN GPU initial implementation";
	}

	public KnnGpuClassifier(CLContext cl_context, int window_size, int k) {
		m_k = k;
		m_window_size = window_size;
		m_cl_context = cl_context;
		m_data_transfer_queue = m_cl_context.createDefaultQueue();
		m_calc_queue = m_cl_context.createDefaultQueue();

		String source = null;

		try {
			source = readKernel("distance.cl");
		} catch (Exception e) {
			e.printStackTrace();
		}
		CLProgram program = m_cl_context.createProgram(source);
		m_distance_kernel = program.createKernel("prepare_distance");
		m_reduction_kernel = program.createKernel("reduction_scalar");
		m_sorter = new BitonicSort(cl_context, m_window_size);
		m_indices = new int[m_window_size];
		for (int i = 0; i < m_indices.length; i++)
			m_indices[i] = i;

		m_workgroup_size = max_workgroup_size();
		m_local_size = (int) Math
				.pow(2,
						Math.floor(Math.log((float) m_workgroup_size)
								/ Math.log(2.0f))); // align to power of 2
	}

	@Override
	public void resetLearningImpl() {
		if (m_window != null) {
			m_window.close(m_data_transfer_queue);
			m_window = null;
		}
	}

	/**
	 * Turn the list of nearest neighbors into a probability distribution.
	 * 
	 * @param neighbours
	 *            the list of nearest neighboring instances
	 * @param distances
	 *            the distances of the neighbors
	 * @return the probability distribution
	 * @throws Exception
	 *             if computation goes wrong or has no class attribute
	 */

	protected double[] makeDistribution(Instance[] neighbours, float[] distances)
			throws Exception {

		double total = 0, weight;
		double[] distribution = new double[m_num_classes];

		// Set up a correction to the estimator
		if (m_class_type == Attribute.NOMINAL) {
			for (int i = 0; i < m_num_classes; i++) {
				distribution[i] = 1.0 / Math.max(1, m_window_size);
			}
			total = (double) m_num_classes / Math.max(1, m_window_size);
		}

		for (int i = 0; i < neighbours.length; i++) {
			// Collect class counts
			Instance current = neighbours[i];

			distances[i] = (float) Math.sqrt(distances[i]
					/ m_num_attributes_used);
			switch (m_distance_weighting) {
			case WEIGHT_INVERSE:
				weight = 1.0 / (distances[i] + 0.001); // to avoid div by zero
				break;
			case WEIGHT_SIMILARITY:
				weight = 1.0 - distances[i];
				break;
			default: // WEIGHT_NONE:
				weight = 1.0;
				break;
			}
			weight *= current.weight();
			try {
				switch (m_class_type) {
				case Attribute.NOMINAL:
					distribution[(int) current.classValue()] += weight;
					break;
				case Attribute.NUMERIC:
					distribution[0] += current.classValue() * weight;
					break;
				}
			} catch (Exception ex) {
				throw new Error("Data has no class attribute!");
			}
			total += weight;
		}

		// Normalise distribution
		if (total > 0) {
			Utils.normalize(distribution, total);
		}
		return distribution;
	}

	private void buildClassifier(Instances instances) {
		m_num_classes = instances.numClasses();
		m_num_attributes = instances.numAttributes();
		m_class_type = instances.classAttribute().type();

		m_window = new HostWindow(m_cl_context, m_window_size,
				m_num_attributes - 1);
		m_output = m_cl_context.createBuffer(Usage.InputOutput, Float.class,
				m_window.getWindowSize());
		m_input = m_cl_context.createBuffer(Usage.Input, Float.class,
				m_window.getInstanceSize());
		m_dist_vector = m_cl_context.createBuffer(Usage.InputOutput,
				Float.class, m_window.getInstanceSize());
		m_num_attributes_used = 0;
		for (int i = 0; i < instances.numAttributes(); i++) {
			if ((i != instances.classIndex())
					&& (instances.attribute(i).isNominal() || instances
							.attribute(i).isNumeric())) {
				m_num_attributes_used += 1;
			}
		}
		
		m_distance_kernel.setArg(0, m_window.getBuffer());
		m_distance_kernel.setArg(1, m_input);
		m_distance_kernel.setArg(2, m_dist_vector);
		
		m_reduction_kernel.setArg(0, m_dist_vector);
		m_reduction_kernel.setArg(1, LocalSize.ofFloatArray(m_local_size));


	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		double[] values = inst.toDoubleArray();

		if (m_window == null) {
			buildClassifier(inst.dataset());
		}
		int classIndex = inst.classIndex();
		int offset = 0;
		float[] array = new float[values.length - 1];
		for (int i = 0; i < values.length; i++)
			if (i != classIndex)
				array[offset++] = (float) values[i];
		try {
			m_last_event = m_window.addInstance(m_data_transfer_queue, inst,
					array);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public synchronized double[] getVotesForInstance(Instance inst) {
		try {
			long time_start = System.currentTimeMillis();
			float[] dists = distance(inst);
			int[] indices = new int[m_window_size];
			System.arraycopy(m_indices, 0, indices, 0, indices.length);
			m_sorter.sort(dists, indices);
			Instance[] neighbours = new Instance[m_k];
			for (int i = 0; i < neighbours.length; i++) {
				neighbours[i] = m_window.instances()[indices[i]];
			}
			double[] res =  makeDistribution(neighbours, dists);
			long time_end = System.currentTimeMillis();
			System.out.println(time_end - time_start);
			return res;

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return new double[0];
	}

	public float[] distance(Instance inst) 
    {
    	if (m_window == null)
    		throw new IllegalArgumentException();
    	CLEvent input_ready = transferInstance(inst, m_input);
    	for (int i = 0 ; i < m_window_size ; i ++ ) {
    		// calculate distance vector
    		int offset = i* m_window.getInstanceSize();
    		m_distance_kernel.setArg(3,  offset);

    		 m_distance_kernel.enqueueNDRange(m_calc_queue,
    				null, 
    				new long[] { m_window.getInstanceSize() }, 
    				null, 
    				input_ready);
    		reduce( i);
    	}
    	m_calc_queue.finish();
    	Pointer<Float> fl = m_output.map(m_calc_queue, MapFlags.Read, new CLEvent[0]);
    	m_output.unmap(m_calc_queue, fl, new CLEvent[0]);
    	return fl.getFloats();
    }

	private void reduce(int i) {
		int global_size = m_window.getInstanceSize();
		int local_size = (int) Math.min(global_size, m_local_size);
		m_reduction_kernel.enqueueNDRange(m_calc_queue, null,
				new long[] { global_size }, new long[] { local_size },
				new CLEvent[0]);

		while (global_size / m_local_size > m_local_size) {
			global_size = (int) (global_size / local_size);
			m_reduction_kernel.enqueueNDRange(m_calc_queue, null,
					new long[] { global_size }, new long[] { local_size },
					new CLEvent[0]);
		}
		
		m_dist_vector.copyTo(m_calc_queue, 0, 1, m_output, i, new CLEvent[0]);	
	}

	private CLEvent transferInstance(Instance inst, CLBuffer<Float> buffer) {
		int classIndex = inst.classIndex();
		int offset = 0;
		double[] values = inst.toDoubleArray();
		float[] array = new float[m_window.getInstanceSize()];
		for (int i = 0; i < values.length; i++)
			if (classIndex != i)
				array[offset++] = (float) values[i];
		Pointer<Float> data = buffer.map(m_data_transfer_queue, MapFlags.Write,
				new CLEvent[0]);
		data.setFloats(array);
		return buffer.unmap(m_data_transfer_queue, data, m_last_event);
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

	@Override
	public boolean isRandomizable() {
		return false;
	}

	public void manageMemory(int currentByteSize, int maxByteSize) {
		// TODO Auto-generated method stub
	}

	public static String readKernel(String name) throws Exception {
		BufferedReader r = new BufferedReader(new InputStreamReader(
				KnnGpuClassifier.class.getResourceAsStream(name)));
		String output = "";
		String line;
		while ((line = r.readLine()) != null)
			output += line + "\n";
		r.close();
		return output;

	}

	private long max_workgroup_size() {
		Map<CLDevice, Long> sizes = m_reduction_kernel.getWorkGroupSize();
		long workgroup_size = Long.MAX_VALUE;
		for (Long l : sizes.values()) {
			if (l < workgroup_size) {
				workgroup_size = l;
			}
		}
		return workgroup_size;
	}

}
