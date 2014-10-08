package org.stream_gpu.knn;

import static org.junit.Assert.*;

import org.junit.Test;

import com.amd.aparapi.device.Device;

public class BitonicSortAparapiTest {

	@Test
	public void test() {
		Device dev = Device.firstGPU();
		
		BitonicSortAparapi sort = new BitonicSortAparapi(dev);
		int[] ind = new int[16];
		float[] values = new float[ind.length];
		for (int i = 0 ; i < ind.length ; i ++)
		{
			ind[i] = i;
			values[i] = ind.length -i;
		}
		
		sort.sort(dev, values, ind);
		assertEquals( ind[0], 15);
	}

}
