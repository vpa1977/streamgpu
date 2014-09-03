package org.stream_gpu.knn.kdtree;

import static org.junit.Assert.*;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

import org.junit.Test;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class TestKdTreeNode {

	
	
	public void setUp()
	{
		
	}
	
	@Test
	public void testRemove() throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.reset();
		loader.setSource(new File("test-remove.arff"));
		Instances instances = loader.getStructure();
		
		
		Instance instTest =loader.getNextInstance(instances); 
		KDTreeWindow window = new KDTreeWindow(16, instances);
		
		Instance inst;
		while ( (inst = loader.getNextInstance(instances)) != null)
		{
			window.add(inst);
		}
		
		int size = window.size();
		assertEquals(size, 16);
		
		ArrayList<GpuInstance> list = window.findNearest(instTest, 4);
		assertEquals(list.size(), 4);

	}

}
