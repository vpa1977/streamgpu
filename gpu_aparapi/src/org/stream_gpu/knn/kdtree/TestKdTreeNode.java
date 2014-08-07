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
		
		KDTreeNode node = new KDTreeNode(instances, null);
		Instance inst = null;
		ArrayList<Instance> list = new ArrayList<Instance>();
		long id = 0;
		while ( (inst = loader.getNextInstance(instances)) != null)
		{
			list.add(inst);
			node.printRanges();
			node.update(new TreeItem(++id,inst), null);
		}
		node.printRanges();
		Collections.reverse(list);
		Iterator<Instance> it = list.iterator();
		while (it.hasNext())
		{
			Instance i = it.next();
			node.update(null, new TreeItem(--id,i));
			node.printRanges();
		}
		 

	}

}
