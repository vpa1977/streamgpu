package org.stream_gpu.knn;

import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.JavaCL;

import moa.classifiers.meta.WEKAClassifier;
import moa.tasks.EvaluatePeriodicHeldOutTest;
import moa.tasks.NullMonitor;
import weka.classifiers.lazy.IBk;

class PrintMontitor extends NullMonitor 
{

	@Override
	public void setCurrentActivity(String activityDescription,
			double fracComplete) {
		// TODO Auto-generated method stub
		super.setCurrentActivity(activityDescription, fracComplete);
		System.out.println(activityDescription + " completed "+ fracComplete);
	}

	@Override
	public void setCurrentActivityDescription(String activity) {
		// TODO Auto-generated method stub
		super.setCurrentActivityDescription(activity);
		System.out.println(activity);
	}
	
}



public class Main {
	public static void main(String[] args){
		
		int k = 20;
		int window = 1000;
		
		int test_size = 10000;
		int train_size = 1000 * 1000;
		
		CLContext context = JavaCL.createContext(null,
				JavaCL.listPlatforms()[0].listAllDevices(false)[0]);

		KnnGpuClassifier classifier = new KnnGpuClassifier(context, window, k);
		EvaluatePeriodicHeldOutTest test = new EvaluatePeriodicHeldOutTest();
		test.learnerOption.setCurrentObject(classifier);
		test.testSizeOption.setValue(test_size);
		test.trainSizeOption.setValue(train_size);
		Object ret = test.doTask(new PrintMontitor(), null);
		System.out.println(ret);
	}
}
