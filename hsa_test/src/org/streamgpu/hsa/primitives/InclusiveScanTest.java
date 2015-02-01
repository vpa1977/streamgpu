package org.streamgpu.hsa.primitives;

import static org.junit.Assert.*;

import org.junit.Test;

import com.amd.aparapi.Device;

public class InclusiveScanTest {

	@Test
	public void test() {
		int[] testval = new int[577];
		int[] resultval = new int[577];
		for (int i = 0 ;i < testval.length ; i ++ )
		{
			testval[i] = 1;
			resultval[i] = i+1;
		}
		
		InclusiveScan scan = new InclusiveScan();
		scan.sum(Device.hsa(), testval);
		assertArrayEquals(resultval, testval);
	}

}
