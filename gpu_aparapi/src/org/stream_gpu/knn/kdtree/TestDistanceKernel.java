package org.stream_gpu.knn.kdtree;

import static org.junit.Assert.*;

import org.junit.Test;

import com.amd.aparapi.Kernel.EXECUTION_MODE;
import com.amd.aparapi.Range;
import com.amd.aparapi.device.Device;

public class TestDistanceKernel {

	@Test
	public void test() {
		DistanceKernel dist = new DistanceKernel(16,5);
		
		dist.m_input_data = new float[16];
		dist.m_results = new float[16];
		
		for (int i = 0 ; i < dist.m_input_data.length ; ++i)
		{
			dist.m_input_data[i] = 1;
		}
		
		for (int i = 0 ;i < 16 ; i ++ )
			dist.m_test[i] = 0;
		
		Device dev = Device.firstGPU();
		//dist.setExecutionMode(EXECUTION_MODE.JTP);
		
		dist.compute(dev, 16);
		
		assertEquals( dist.m_results[0], 16 , 0);
	}

}
