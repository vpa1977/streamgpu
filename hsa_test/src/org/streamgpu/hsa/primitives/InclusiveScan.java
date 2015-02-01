package org.streamgpu.hsa.primitives;

import com.amd.aparapi.Aparapi.IntTerminal;
import com.amd.aparapi.Device;

public class InclusiveScan {
	public void sum(Device dev, int[] values)
	{
		IntTerminal lambda = k -> {
			values[k] = 1;
		};
		
		dev.forEach(values.length, lambda);
	}
}




