package org.stream_gpu.knn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.util.Arrays;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;

import org.bridj.Pointer;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem.MapFlags;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLQueue;

public class KnnGpuClassifier extends  AbstractClassifier {

    private static final long serialVersionUID = 1L;
    private HostWindow m_window;
    private int m_k;
    private CLQueue m_data_transfer_queue;
    private CLQueue m_calc_queue;
    private CLContext m_cl_context;
    private CLEvent m_last_event;
    private CLKernel m_distance_kernel;
    private CLBuffer<Float> m_output;
    private CLBuffer<Float> m_input;
    private int m_window_size;
    private int m_num_classes;
	private int m_num_attributes;
	private int m_class_type;

    @Override
    public String getPurposeString() {
        return "kNN GPU initial implementation";
    }
    
    public KnnGpuClassifier( CLContext cl_context, int window_size, int k )
    {
    	m_k = k;
    	m_window_size = window_size;
    	m_cl_context = cl_context;
    	m_data_transfer_queue = m_cl_context.createDefaultQueue();
    	m_calc_queue = m_cl_context.createDefaultQueue();
    	
    	String source = null;
    	
    	try {
    		source = readKernel("distance.cl");
    	} catch (Exception e){ e.printStackTrace();}
		m_distance_kernel = m_cl_context.createProgram(source).createKernel("square_distance");
		
		
    }
    

    @Override
    public void resetLearningImpl() {
    	if (m_window!= null) {
    		m_window.close(m_data_transfer_queue);
    		m_window = null;
    	}
    }
    
    
    /**
     * Turn the list of nearest neighbors into a probability distribution.
     *
     * @param neighbours the list of nearest neighboring instances
     * @param distances the distances of the neighbors
     * @return the probability distribution
     * @throws Exception if computation goes wrong or has no class attribute
     */
/*    
    protected double [] makeDistribution(Instances neighbours, double[] distances)
      throws Exception {

      double total = 0, weight;
      double [] distribution = new double [m_num_classes];
      
      // Set up a correction to the estimator
      if (m_ClassType == Attribute.NOMINAL) {
        for(int i = 0; i < m_NumClasses; i++) {
  	distribution[i] = 1.0 / Math.max(1,m_Train.numInstances());
        }
        total = (double)m_NumClasses / Math.max(1,m_Train.numInstances());
      }

      for(int i=0; i < neighbours.numInstances(); i++) {
        // Collect class counts
        Instance current = neighbours.instance(i);
        distances[i] = distances[i]*distances[i];
        distances[i] = Math.sqrt(distances[i]/m_NumAttributesUsed);
        switch (m_DistanceWeighting) {
          case WEIGHT_INVERSE:
            weight = 1.0 / (distances[i] + 0.001); // to avoid div by zero
            break;
          case WEIGHT_SIMILARITY:
            weight = 1.0 - distances[i];
            break;
          default:                                 // WEIGHT_NONE:
            weight = 1.0;
            break;
        }
        weight *= current.weight();
        try {
          switch (m_ClassType) {
            case Attribute.NOMINAL:
              distribution[(int)current.classValue()] += weight;
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
*/

    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	double[] values = inst.toDoubleArray();
    	
    	if (m_window == null) {
    		m_num_classes = inst.dataset().numClasses();
    		m_num_attributes = inst.dataset().numAttributes();
    		m_class_type = inst.dataset().classAttribute().type();
    		
    		m_window = new HostWindow(m_cl_context, m_window_size, m_num_attributes - 1);
    		m_output = m_cl_context.createBuffer(Usage.InputOutput,Float.class, m_window.getWindowSize());
    		m_input = m_cl_context.createBuffer(Usage.Input, Float.class, m_window.getInstanceSize());
    	}
    	int classIndex = inst.classIndex();
    	int offset =0;
    	float[] array = new float[ values.length - 1];
    	for (int i = 0 ; i < values.length ; i ++ ) 
    		if (i != classIndex)
    			array[offset++] = (float)values[i];
    	try {
			m_last_event = m_window.addInstance(m_data_transfer_queue,inst, array);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
    
    public float[] sort(float[] input)
    {
    	Arrays.sort(input);
    	float[] dest = new float[m_k];
    	System.arraycopy(input, 0, dest, 0, m_k);
    	return dest;
    }
    
    

    @Override
    public synchronized double[] getVotesForInstance(Instance inst) {
    	try {
			//return makeDistribution(sort(distance(inst)));
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
    	
		m_distance_kernel.setArgs(
						 m_input,
						 m_window.getBuffer(),
						 m_output, 
						 m_window.getInstanceSize());

    	CLEvent distance_done = m_distance_kernel.enqueueNDRange(m_calc_queue,
    			 			null, // offsets
    						new long[] { m_window.getWindowSize() }, // global sizes 
    						null, // local sizes
    			 new CLEvent[]{ m_last_event , input_ready} ); 
    	
		Pointer<Float> result = m_output.map(m_calc_queue, MapFlags.Read,	new CLEvent[]{distance_done});
		float[] result_array = result.getFloats();
		//result.setFloats(new float[result_array.length]);
		m_output.unmap(m_calc_queue, result , new CLEvent[0]);
		m_calc_queue.finish();
		return result_array;
    }

	private CLEvent transferInstance(Instance inst, CLBuffer<Float> buffer) {
		int classIndex = inst.classIndex();
    	int offset =0;
		double[] values = inst.toDoubleArray();
    	float[] array = new float[ m_window.getInstanceSize() ];
    	for (int i = 0 ; i < values.length ; i ++ )
    		if (classIndex != i)
    		array[offset++] = (float)values[i];
    	Pointer<Float> data = buffer.map(m_calc_queue, MapFlags.Write, new CLEvent[0]);
    	data.setFloats(array);
    	return buffer.unmap(m_calc_queue, data, new CLEvent[0]);
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
		BufferedReader r = new BufferedReader(new InputStreamReader(KnnGpuClassifier.class.getResourceAsStream(name)));
		String output = "";
		String line;
		while ((line = r.readLine()) != null)
			output += line + "\n";
		r.close();
		return output;

	}

}
