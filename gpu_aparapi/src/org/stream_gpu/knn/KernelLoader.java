package org.stream_gpu.knn;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class KernelLoader {

	static String readKernel(String name) throws Exception {
		BufferedReader r = new BufferedReader(new InputStreamReader(KnnGpuClassifier.class.getResourceAsStream(name)));
		String output = "";
		String line;
		while ((line = r.readLine()) != null)
			output += line + "\n";
		r.close();
		return output;
	
	}

}
