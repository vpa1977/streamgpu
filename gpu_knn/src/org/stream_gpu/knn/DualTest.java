package org.stream_gpu.knn;

import org.stream_gpu.float_knn.float_search.LinearNNSearch_Float;

import weka.classifiers.lazy.IBk;
import moa.classifiers.meta.WEKAClassifier;
import moa.streams.generators.RandomTreeGenerator;
import moa.tasks.EvaluatePeriodicHeldOutTest;
import moa.tasks.NullMonitor;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.JavaCL;

public class DualTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		System.out.println( System.getProperty("java.library.path"));
		
		int k = 32;
		int window = 8192*2*2*2*2;
		
		int test_size  = 1000;
		int train_size = 1000000;
		
		
		
		RandomTreeGenerator generator = new RandomTreeGenerator();
		
		//generator.numNumericsOption.setValue(128);
		//generator.numNominalsOption.setValue(128);
		CLDevice device = JavaCL.listPlatforms()[0].listAllDevices(false)[0];
		System.out.println(device.getName());
		CLContext context = JavaCL.createContext(null,
				device);

		
		KnnGpuClassifier classifier = new KnnGpuClassifier(context, window, k);
		
		IBk kMeans = new IBk(k);
		kMeans.setNearestNeighbourSearchAlgorithm(new LinearNNSearch_Float());
		kMeans.setWindowSize(window);
		
		
		WEKAClassifier wekaClassifier = new WEKAClassifier();
		wekaClassifier.baseLearnerOption.setCurrentObject(kMeans);
		System.out.println("Test : window="+ window + " k =" + k);
		System.out.println("	 : test_size="+ test_size + " train_size =" + train_size);
		System.out.println("Stream: "+generator.getClass() + " "+ generator.getOptions().getAsCLIString());
		
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
