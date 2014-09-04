package org.stream_gpu.knn;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;

import org.bridj.Pointer;
import org.stream_gpu.knn.kdtree.GpuInstance;
import org.stream_gpu.knn.kdtree.KDTreeNode;
import org.stream_gpu.knn.kdtree.KDTreeWindow;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLMem;
import com.nativelibs4java.opencl.CLQueue;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class KnnKdTreeGpuClassifier  extends AbstractClassifier {
	
    private static final long serialVersionUID = 1L;
    /** no weighting. */
    public static final int WEIGHT_NONE = 1;
    /** weight by 1/distance. */
    public static final int WEIGHT_INVERSE = 2;
    /** weight by 1-distance. */
    public static final int WEIGHT_SIMILARITY = 4;

    private KDTreeWindow m_window;
    
	private IDistance m_distance;
	private BitonicSort m_sorter;
	private int m_num_attributes_used;
	private int m_distance_weighting;
	private int m_k;
    private int m_window_size;
    private int m_num_classes;
    private int m_class_type;

    @Override
    public String getPurposeString() {
        return "kNN GPU initial implementation";
    }
    
    public KnnKdTreeGpuClassifier(int window_size, int k )
    {
    	m_k = k;
    	m_window_size = window_size;
		m_distance_weighting = WEIGHT_NONE;
		
    }
    

    @Override
    public void resetLearningImpl() {
    	if (m_window!= null) {
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
    
    protected double [] makeDistribution(ArrayList<GpuInstance> instances)
      throws Exception {

      double total = 0, weight;
      double [] distribution = new double [m_num_classes];
      
      // Set up a correction to the estimator
      if (m_class_type == Attribute.NOMINAL) {
        for(int i = 0; i < m_num_classes; i++) {
        	distribution[i] = 1.0 / Math.max(1,m_window_size);
        }
        total = (double)m_num_classes / Math.max(1,m_window_size);
      }

      for(GpuInstance gpu_instance : instances) {
        // Collect class counts
        Instance current = gpu_instance.wekaInstance();
        double distance = gpu_instance.distance();
        switch (m_distance_weighting) {
          case WEIGHT_INVERSE:
            weight = 1.0 / (Math.sqrt(distance/m_num_attributes_used) + 0.001); // to avoid div by zero
            break;
          case WEIGHT_SIMILARITY:
            weight = 1.0 - Math.sqrt(distance/m_num_attributes_used);
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
		m_num_attributes_used = 0;
        for (int i = 0; i < instances.numAttributes(); i++) {
          if ((i != instances.classIndex()) && 
    	  (instances.attribute(i).isNominal() || instances.attribute(i).isNumeric())) {
        	  m_num_attributes_used += 1;
          }
        }
        m_window = new KDTreeWindow(m_window_size, instances);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	if (m_window == null) {
    		buildClassifier( inst.dataset());
    	}
    	m_window.add(inst);
    }
    
   

    @Override
    public synchronized double[] getVotesForInstance(Instance inst) {
    	try {
    		ArrayList<GpuInstance> nodes_to_check = m_window.findNearest(inst, m_k);
    		
    		return makeDistribution(nodes_to_check);
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
    

}
