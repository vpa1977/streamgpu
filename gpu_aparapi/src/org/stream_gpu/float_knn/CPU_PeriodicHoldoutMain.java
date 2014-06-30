package org.stream_gpu.float_knn;

import org.stream_gpu.float_knn.float_search.LinearNNSearch_Float;

import weka.classifiers.lazy.IBk;
import moa.classifiers.meta.WEKAClassifier;
import moa.streams.generators.RandomTreeGenerator;
import moa.tasks.EvaluatePeriodicHeldOutTest;
import moa.tasks.NullMonitor;

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



public class CPU_PeriodicHoldoutMain {
	

	public static void main(String[] args){
		
		int k = 16;
		int window = 256;
		
		int test_size = 10000;
		int train_size = 1000 * 1000;
				
		IBk kMeans = new IBk(k);
		kMeans.setNearestNeighbourSearchAlgorithm(new LinearNNSearch_Float());
		kMeans.setWindowSize(window);
		
		WEKAClassifier wekaClassifier = new WEKAClassifier();
		wekaClassifier.baseLearnerOption.setCurrentObject(kMeans);

		RandomTreeGenerator generator = new RandomTreeGenerator(); 
		
		EvaluatePeriodicHeldOutTest test = new EvaluatePeriodicHeldOutTest();
		test.learnerOption.setCurrentObject(wekaClassifier);
		test.testSizeOption.setValue(test_size);
		test.trainSizeOption.setValue(train_size);
		test.streamOption.setCurrentObject(generator);
		
		Object ret = test.doTask(new PrintMontitor(), null);
		System.out.println(ret);
	}
}
