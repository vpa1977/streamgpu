package hsa_test;

import static org.junit.Assert.*;

import java.util.ArrayList;

import org.junit.Test;

import com.amd.aparapi.Device;

public class InstanceDistanceTest {

	@Test
	public void test() throws Throwable {
		InstanceDistance dist = new InstanceDistance();
		
		int sample_size = 256;
		dist.loadOkra();
		dist.loadOcl();
		do {
			System.out.println("Sample Size "+ sample_size);
			float[] val1 = new float[16];
			float[] val2 = new float[16];
			
			
			for (int i = 0 ;i < 16 ; i ++)
			{
				val1[i] = 1;
				val2[i] = 0;
			}
			
			
			float[][] samples = new float[sample_size][];
			float[] flat_samples = new float[ sample_size * 16];
		
			for (int i = 0 ; i < sample_size ; ++i)
			{
				samples[i] =new float[16];
				System.arraycopy(val2, 0, samples[i], 0, 16);
				System.arraycopy(val2, 0, flat_samples, i*16, 16);
			}
			
			
			Device dev = Device.hsa();
			dist.distance_hsa(dev,samples, val2, 10, 16);
			for (int i = 0 ;i < 10000 ; i ++ )
				dist.distance_hsa(dev,samples, val2, 10, 16);
			
			
			
			long start = System.nanoTime();
			for (int i = 0 ;i < 10000 ; i ++ )
					dist.distance_hsa(dev,samples, val2, 10, 16);
			long end = System.nanoTime();
			System.out.println("Hsa:" + (end - start)/1000000);
	
			
					
			dev = Device.seq();
			for (int i = 0 ;i < 10000 ; i ++ )
				dist.distance_hsa(dev,samples, val2, 10, 16);
		
			start = System.nanoTime();
			for (int i = 0 ;i < 10000 ; i ++ )
					dist.distance_hsa(dev,samples, val2, 10, 16);
			end = System.nanoTime();
			System.out.println("Seq:" + (end - start)/1000000);
			
			
			for (int i = 0 ;i < 10000 ; i ++ )
				dist.distance_okra(flat_samples, val2);
		
			start = System.nanoTime();
			for (int i = 0 ;i < 10000 ; i ++ )
					dist.distance_okra(flat_samples, val2);
			end = System.nanoTime();
			System.out.println("Okra:" + (end - start)/1000000);
			
			
			sample_size *= 2;
		} while (sample_size < 128000);

	}

}
