package org.stream_gpu.knn.kdtree.test;

import static org.junit.Assert.*;

import java.io.File;
import java.util.ArrayList;

import org.junit.Before;
import org.junit.Test;
import org.stream_gpu.knn.kdtree.GpuInstance;
import org.stream_gpu.knn.kdtree.GpuInstances;

import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;


public class TestRange {
	
	private ArrayList<Instance> m_data;
	private GpuInstances m_gpu_model;
	private Instances m_weka_model;
	@Before
	public void setUp() throws Throwable
	{
		ArffLoader loader = new ArffLoader();
		loader.reset();
		loader.setSource(new File("elecNormNew.arff"));
		
		Instances instances = loader.getStructure();
		GpuInstances gpuModel = new GpuInstances(instances);
		
		ArrayList<Instance> loaderData = new ArrayList<Instance>();
		Instance inst;
		while ( (inst = loader.getNextInstance(instances)) != null)
		{
			loaderData.add(loader.getNextInstance(instances));
		}
		m_data = loaderData;
		m_gpu_model = gpuModel;
		m_weka_model = instances;
	}
	
	
	public void testCPUSpeed()
	{
		long numIter = 10000;
		double totalTime = 0;
		for (int i = 0 ;i < numIter ; i ++)
		{
			EuclideanDistance dist = new EuclideanDistance(m_weka_model);
			long start = System.nanoTime();
			for (Instance inst : m_data)
			{
				dist.updateRanges(inst);
			}
			long end = System.nanoTime();
			double elapsed =(end-start)/1000000;
			assertTrue(elapsed > 0);
			totalTime += elapsed;
			
		}
		System.out.println("CPU done in " + (totalTime / numIter) );
	}

	@Test
	public void testRangeUpdateSpeed() 
	{
		System.out.println("Call started");
		
		long numIter = 100;
		double totalTime = 0;
		for (int i = 0 ;i < numIter ; i ++)
		{
			long start = System.nanoTime();
			for (Instance inst : m_data)
			{
				GpuInstance gpuInst = m_gpu_model.createInstance(inst);
				
			}
				
			long end = System.nanoTime();
			double elapsed =(end-start)/1000000;
			assertTrue(elapsed > 0);
			totalTime += elapsed;
			System.out.println("Stage "+i);
		}
		System.out.println("CPU done in " + (totalTime / numIter) );
	}
	
	public static void main(String[] string) throws Throwable
	{
		System.setProperty("com.amd.aparapi.enableVerboseJNI","true");
		System.setProperty("com.amd.aparapi.enableVerboseJNIOpenCLResourceTracking","true");

		TestRange r = new TestRange();
		r.setUp();
		r.testRangeUpdateSpeed();
	
	}
}
