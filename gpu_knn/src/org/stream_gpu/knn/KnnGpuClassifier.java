package org.stream_gpu.knn;

import java.io.BufferedReader;
import java.io.FileReader;

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
    private CLKernel m_kernel;
    private CLBuffer<Float> m_output;
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
    		source = readKernel();
    	} catch (Exception e){ e.printStackTrace();}
		m_kernel = m_cl_context.createProgram(source).createKernel("knn");
		
		
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
    protected double [] makeDistribution(float[] distances)
      throws Exception {

      double total = 0, weight =  1.0;
      double [] distribution = new double [m_num_classes];
      
      total = (double)m_num_classes / Math.max(1, m_window_size);
      
      for(int i=0; i < m_window_size; i++) {
        // Collect class counts
        distances[i] = distances[i]*distances[i];
        distances[i] = (float)Math.sqrt(distances[i]/m_num_attributes);
        weight *= m_window.instances()[i].weight();
        try {
          switch (m_class_type) {
            case Attribute.NOMINAL:
              distribution[(int)m_window.instances()[i].classValue()] += weight;
              break;
            case Attribute.NUMERIC:
              distribution[0] += m_window.instances()[i].classValue() * weight;
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


    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	double[] values = inst.toDoubleArray();
    	
    	if (m_window == null) {
    		m_window = new HostWindow(m_cl_context, m_window_size, values.length);
    		m_output = m_cl_context.createBuffer(Usage.Output,Float.class, m_window.getInstanceSize());
    		m_num_classes = inst.dataset().numClasses();
    		m_num_attributes = inst.dataset().numAttributes();
    		m_class_type = inst.dataset().classAttribute().type();
    	}
    	
    	float[] array = new float[ values.length];
    	for (int i = 0 ; i < values.length ; i ++ ) 
    		array[i] = (float)values[i];
    	m_last_event = m_window.addInstance(m_data_transfer_queue,inst, array);
    }

    @Override
    public synchronized double[] getVotesForInstance(Instance inst) {
    	if (m_window == null)
    		throw new IllegalArgumentException();
		m_kernel.setArgs(m_window.getBuffer(),
						 m_window.getInstanceSize(), 
						 m_window.getWindowSize(),
						 m_output);

    	m_kernel.enqueueNDRange(m_calc_queue, 
    						new long[] { 0, 0 },
    						new long[] { m_window.getInstanceSize(),
    			 m_window.getWindowSize() }, null
    			, m_last_event);
    	m_calc_queue.finish();
		Pointer<Float> result = m_output.map(m_calc_queue, MapFlags.Read,	new CLEvent[0]);
		float[] result_array = result.getFloats();
		m_output.unmap(m_calc_queue, result, new CLEvent[0]);
    	try {
			return makeDistribution(result_array);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	return new double[0];
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
    
    

	public static String readKernel() throws Exception {
		BufferedReader r = new BufferedReader(new FileReader("knn.cl"));
		String output = "";
		String line;
		while ((line = r.readLine()) != null)
			output += line + "\n";
		r.close();
		return output;

	}

}
