package org.stream_gpu.knn;

import java.io.BufferedReader;
import java.io.InputStreamReader;

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
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.CLQueue;

public class KnnGpuClassifier extends  AbstractClassifier {

    private static final long serialVersionUID = 1L;
    /** no weighting. */
    public static final int WEIGHT_NONE = 1;
    /** weight by 1/distance. */
    public static final int WEIGHT_INVERSE = 2;
    /** weight by 1-distance. */
    public static final int WEIGHT_SIMILARITY = 4;

    private SlidingWindow m_window;
    private int m_k;
    private CLQueue m_data_transfer_queue;
    private CLQueue m_calc_queue;
    private CLContext m_cl_context;
    private CLEvent m_last_event;
    private int m_window_size;
    private int m_num_classes;
	private int m_class_type;
	
	
	private IDistance m_distance;
	private BitonicSort m_sorter;
	private int m_num_attributes_used;
	private int m_distance_weighting;

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
		m_distance_weighting = WEIGHT_NONE;
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
    
    protected double [] makeDistribution(int[] indices, CLBuffer<Float> distance_buffer)
      throws Exception {

      double total = 0, weight;
      double [] distribution = new double [m_num_classes];
      float[] distances = null;
      
      if (m_distance_weighting != WEIGHT_NONE)
      {
    	 distances = new float[ indices.length];
    	 Pointer<Float> mapped = distance_buffer.map(m_sorter.getQueue(),  CLMem.MapFlags.Read, new CLEvent[0]);
    	 mapped.getFloats(distances);
    	 distance_buffer.unmap(m_sorter.getQueue(), mapped, new CLEvent[0]);
      }
      
      // Set up a correction to the estimator
      if (m_class_type == Attribute.NOMINAL) {
        for(int i = 0; i < m_num_classes; i++) {
        	distribution[i] = 1.0 / Math.max(1,m_window_size);
        }
        total = (double)m_num_classes / Math.max(1,m_window_size);
      }

      for(int i=0; i < m_k; i++) {
        // Collect class counts
        Instance current = m_window.instances()[indices[i]];
        switch (m_distance_weighting) {
          case WEIGHT_INVERSE:
            weight = 1.0 / (Math.sqrt(distances[i]/m_num_attributes_used) + 0.001); // to avoid div by zero
            break;
          case WEIGHT_SIMILARITY:
            weight = 1.0 - Math.sqrt(distances[i]/m_num_attributes_used);
            break;
          default:                                 // WEIGHT_NONE:
            weight = 1.0;
            break;
        }
        weight *= current.weight();
        try {
          switch (m_class_type) {
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
    
    private void buildClassifier( Instances instances)
    {
		m_num_classes = instances.numClasses();
		m_class_type = instances.classAttribute().type();
		
		m_window = new SlidingWindow(m_cl_context, m_window_size, instances);
		m_distance = new Distance(m_window, m_calc_queue, m_cl_context);
		m_sorter = new BitonicSort(m_cl_context, m_window_size); 
		m_num_attributes_used = 0;
        for (int i = 0; i < instances.numAttributes(); i++) {
          if ((i != instances.classIndex()) && 
    	  (instances.attribute(i).isNominal() || instances.attribute(i).isNumeric())) {
        	  m_num_attributes_used += 1;
          }
        }

    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	if (m_window == null) {
    		buildClassifier( inst.dataset());
    	}
		m_window.addInstance(m_data_transfer_queue,inst);
    }
    
   

    @Override
    public synchronized double[] getVotesForInstance(Instance inst) {
    	try {
    		if (!m_window.ready())
    			throw new IllegalArgumentException("Not enough values in the sliding window");
    		m_last_event = m_window.flushInstances(m_data_transfer_queue);
    		CLBuffer<Float> distances = distance(inst);
    		int[] indices = new int[m_k];
    		m_sorter.sort(distances, indices);
    		return makeDistribution(indices, distances);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	return new double[0];
    }
    
    public CLBuffer<Float> distance(Instance inst) 
    {
    	if (m_window == null)
    		throw new IllegalArgumentException();
		return m_distance.distance(inst,m_last_event);
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
