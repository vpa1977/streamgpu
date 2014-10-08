package org.amd.sdk;

import static org.junit.Assert.*;

import org.junit.Test;

import com.amd.aparapi.device.Device;

public class BitonicSortTest {

	@Test
	public void test() {
		BitonicSort sort = new BitonicSort();
		
		int[] ints = new int[23];
		float[] to_sort = new float[23];
		for (int i = 0 ;i < to_sort.length ; i ++ )
		{
			to_sort[i] = 23-i;
			ints[i] = i;
		}
		
		sort.sort(Device.firstGPU(), to_sort, ints);
		
		System.out.println(to_sort[0]);
		
	}

}
