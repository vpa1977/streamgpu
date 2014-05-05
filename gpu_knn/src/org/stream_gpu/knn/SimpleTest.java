package org.stream_gpu.knn;

import java.util.ArrayList;

import javax.swing.JFrame;

import org.stream_gpu.float_knn.float_search.EuclideanDistance;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.JavaCL;

public class SimpleTest {
	
	
	public double distance( Instance src, Instance dst)
	{
		double[] src_val = src.toDoubleArray();
		double[] dst_val = dst.toDoubleArray();
		int classIndex = src.classIndex();
		double result = 0;
		for (int i= 0;i< src_val.length ; i ++ )
		{
			if (i == classIndex)
				continue;
			result += (src_val[i] - dst_val[i])*(src_val[i] - dst_val[i]);
		}
		return result;
	}
	
	public double[] distance(Instance src, Instance[] samples)
	{
		double[] dist = new double[samples.length];
		for (int i = 0 ;i < samples.length ; i ++ )
		{
			dist[i] = distance(src, samples[i]);
		}
		return dist;
	}
	
	
	public static void main(String[] args)
	{
		
		System.out.println(System.getProperty("java.class.path"));
		CLContext context = JavaCL.createContext(null,
				JavaCL.listPlatforms()[0].listAllDevices(false)[0]);

		KnnGpuClassifier clazz = new KnnGpuClassifier(context, 2, 2);
		
		ArrayList<Attribute> attinfo = new ArrayList<Attribute>();
		attinfo.add( new Attribute("1"));
		attinfo.add( new Attribute("2"));
		attinfo.add( new Attribute("3"));
		attinfo.add( new Attribute("4"));
		
		Instances dataset = new Instances( "test", attinfo , 1);
		dataset.setClassIndex(0);
		Instance inst = new DenseInstance(1, new double[]{ 1, 1, 1 , 1 });
		Instance inst1 = new DenseInstance(1, new double[]{ 2, 2, 2, 2});
		inst.setDataset(dataset);
		inst1.setDataset(dataset);
		
		clazz.trainOnInstance(inst);
		clazz.trainOnInstance(inst1);
		
		
		
		CLBuffer<Float> dist = clazz.distance(inst);
		
		
		float[] to_be_sorted = new float[] {16,15,14,22,12,11,10,9,8,7,6,5,4,3,2,1};
		int[] indices = new int[]{ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
		
		BitonicSort sort = new BitonicSort(context , to_be_sorted.length);
		
		//float[] result = sort.sort(to_be_sorted, indices);
		//loat []  dist1 =  clazz.distance(inst1);
		
		EuclideanDistance float_dist = new EuclideanDistance(dataset);
		float_dist.setDontNormalize(true);
		
		float d1 = float_dist.distance(inst, inst1, Float.MAX_VALUE);
		float d2 = float_dist.distance(inst, inst, Float.MAX_VALUE);
		
		System.out.println();
		new JFrame().setVisible(true);
		
	}
}
