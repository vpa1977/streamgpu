package org.amd.sdk;

import java.util.Arrays;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Range;
import com.amd.aparapi.device.Device;
/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

public class BitonicSort extends Kernel{
	
	public BitonicSort()
	{
		setExplicit(true);
	}
	
	float[] theArray;
	int[] theIndices;
	int stage;
	int direction;
	
	public int nextPow2(int v ) 
	{
		v--;
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v++;
		return v;
	}
	
	public void sort(Device d, float[] data, int[] indices)
	{
		int length = nextPow2(data.length);
		theArray = new float[length];
	    System.arraycopy(data, 0, theArray, 0, data.length);
	    Arrays.fill(theArray, data.length , length, Integer.MAX_VALUE);
	    
	    theIndices = new int[length];
	    System.arraycopy(indices, 0, theIndices, 0, indices.length);
	    Arrays.fill(theIndices, indices.length , length, Integer.MAX_VALUE);
	    
	    put(theArray);
	    put(theIndices);
	    
	    int numStages = 0;
	    int temp;
	    
	    
	    int globalWidth = length/2;
	    Range r = d.createRange(globalWidth); 
	    /*
	     * This algorithm is run as NS stages. Each stage has NP passes.
	     * so the total number of times the kernel call is enqueued is NS * NP.
	     *
	     * For every stage S, we have S + 1 passes.
	     * eg: For stage S = 0, we have 1 pass.
	     *     For stage S = 1, we have 2 passes.
	     *
	     * if length is 2^N, then the number of stages (numStages) is N.
	     * Do keep in mind the fact that the algorithm only works for
	     * arrays whose size is a power of 2.
	     *
	     * here, numStages is N.
	     *
	     * For an explanation of how the algorithm works, please go through
	     * the documentation of this sample.
	     */

	    /*
	     * 2^numStages should be equal to length.
	     * i.e the number of times you halve length to get 1 should be numStages
	     */
	    for(temp = length; temp > 1; temp >>= 1)
	    {
	        ++numStages;
	    }

	    // Set appropriate arguments to the kernel

	    // the input array - also acts as output for this pass (input for next)
	    direction = 1;


	    for(stage = 0; stage < numStages; ++stage)
	    {
	    	this.execute(r, stage+1);
	    }
	    System.arraycopy(theArray, 0, data, 0, data.length);
	    System.arraycopy(theIndices, 0, indices, 0, indices.length);
	}
	
	public void run() 
	{
	    int sortIncreasing = direction;
	    int threadId = getGlobalId(0);
	    
	    int pairDistance = 1 << (stage - this.getPassId());
	    int blockWidth   = 2 * pairDistance;

	    int leftId = (threadId % pairDistance) 
	                   + (threadId / pairDistance) * blockWidth;

	    int rightId = leftId + pairDistance;
	    
	    float leftElement = theArray[leftId];
	    float rightElement = theArray[rightId];
	    
	    int left_idx  = theIndices[leftId];
		int right_idx = theIndices[rightId];
	    
	    int sameDirectionBlockWidth = 1 << stage;
	    
	    if((threadId/sameDirectionBlockWidth) % 2 == 1)
	        sortIncreasing = 1 - sortIncreasing;

	    
	    if(leftElement > rightElement)
	    {
		    if(sortIncreasing > 0)
		    {
		        theArray[leftId]  = rightElement;
		        theIndices[leftId] = right_idx;
		        
		        theArray[rightId] = leftElement;
		        theIndices[rightId] = left_idx;
		    }
		    else
		    {
		        theArray[leftId]  = leftElement;
		        theIndices[leftId] = left_idx;
		        
		        theArray[rightId] = rightElement;
		        theIndices[rightId] = right_idx;
		    }
	    }
	    else
	    {
		    if(sortIncreasing > 0)
		    {
		        theArray[leftId]  = leftElement;
		        theIndices[leftId]  = left_idx;
		        
		        theArray[rightId] = rightElement;
		        theIndices[rightId] = right_idx;
		    }
		    else
		    {
		        theArray[leftId]  = rightElement;
		        theIndices[leftId] = right_idx;
		        
		        theArray[rightId] = leftElement;
		        theIndices[rightId] = left_idx;
		    }
	        
	    }
	}

}
