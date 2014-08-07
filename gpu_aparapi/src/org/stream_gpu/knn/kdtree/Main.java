package org.stream_gpu.knn.kdtree;

import java.io.File;
import java.io.IOException;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class Main {

	public static void main(String[] args) throws Throwable
	{
		int window_size = 32;
		ArffLoader loader = new ArffLoader();
		loader.reset();
		loader.setSource(new File("elecNormNew.arff"));
		
		Instances instances = loader.getStructure();
		
		KDTreeWindow window = new KDTreeWindow(window_size, instances);
		Instance inst = null;
		while ( (inst = loader.getNextInstance(instances)) != null)
		{
			window.add(loader.getNextInstance(instances));
		}
		System.out.println(window.size());
	}
}
