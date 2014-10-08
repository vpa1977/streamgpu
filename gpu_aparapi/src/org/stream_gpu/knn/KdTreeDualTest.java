package org.stream_gpu.knn;

import org.stream_gpu.float_knn.float_search.LinearNNSearch_Float;

import weka.classifiers.lazy.IBk;
import weka.core.neighboursearch.KDTree;
import moa.classifiers.meta.WEKAClassifier;
import moa.streams.InstanceStream;
import moa.streams.generators.RandomTreeGenerator;
import moa.tasks.EvaluatePeriodicHeldOutTest;
import moa.tasks.NullMonitor;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.JavaCL;

public class KdTreeDualTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Throwable {
		
		//System.setProperty("com.amd.aparapi.enableVerboseJNI","true");
		//System.setProperty("com.amd.aparapi.enableVerboseJNIOpenCLResourceTracking","true");

		
		int k = Integer.parseInt(args[0]);
		int window = Integer.parseInt(args[1]);
		
		int test_size  = Integer.parseInt(args[2]);
		int train_size = Integer.parseInt(args[3]);
		
		
		
		InstanceStream generator = (InstanceStream)Class.forName(args[4]).newInstance();
		
		KnnKdTreeGpuClassifier classifier = new KnnKdTreeGpuClassifier(window, k);
		
		IBk kMeans = new IBk(k);
		kMeans.setNearestNeighbourSearchAlgorithm(new KDTree());
		kMeans.setWindowSize(window);
		
		
		WEKAClassifier wekaClassifier = new WEKAClassifier();
		wekaClassifier.baseLearnerOption.setCurrentObject(kMeans);
		System.out.println("Test : window="+ window + " k =" + k);
		System.out.println("	 : test_size="+ test_size + " train_size =" + train_size);
		System.out.println("Stream: "+generator.getClass());// + " "+ generator.getOptions().getAsCLIString());
		
		EvaluatePeriodicHeldOutTest test = new EvaluatePeriodicHeldOutTest();
		test.streamOption.setCurrentObject(generator);
		test.learnerOption.setCurrentObject(classifier);
		test.testSizeOption.setValue(test_size);
		test.trainSizeOption.setValue(train_size);
		System.out.println("------Classifier:"+ test.learnerOption.getValueAsCLIString()  );
		Object ret = test.doTask(new NullMonitor(),null);
		System.out.println(ret);
		System.out.println("---------------------------------------------------------------------------");
		
		test = new EvaluatePeriodicHeldOutTest();
		test.learnerOption.setCurrentObject(wekaClassifier);
		test.streamOption.setCurrentObject(generator);
		test.testSizeOption.setValue(test_size);
		test.trainSizeOption.setValue(train_size);
		System.out.println("------Classifier:"+ test.learnerOption.getValueAsCLIString()  );
		ret = test.doTask(new NullMonitor(),null);
		System.out.println(ret);
		System.out.println("---------------------------------------------------------------------------");
		
	}

}
