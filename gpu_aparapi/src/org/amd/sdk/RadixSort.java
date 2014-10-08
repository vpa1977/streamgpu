package org.amd.sdk;

import java.util.Arrays;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Range;
import com.amd.aparapi.device.Device;
/* ============================================================

Copyright (c) 2009-2010 Advanced Micro Devices, Inc.  All rights reserved.
 
Redistribution and use of this material is permitted under the following 
conditions:
 
Redistributions must retain the above copyright notice and all terms of this 
license.
 
In no event shall anyone redistributing or accessing or using this material 
commence or participate in any arbitration or legal action relating to this 
material against Advanced Micro Devices, Inc. or any copyright holders or 
contributors. The foregoing shall survive any expiration or termination of 
this license or any agreement or access or use related to this material. 

ANY BREACH OF ANY TERM OF THIS LICENSE SHALL RESULT IN THE IMMEDIATE REVOCATION 
OF ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE THIS MATERIAL.

THIS MATERIAL IS PROVIDED BY ADVANCED MICRO DEVICES, INC. AND ANY COPYRIGHT 
HOLDERS AND CONTRIBUTORS "AS IS" IN ITS CURRENT CONDITION AND WITHOUT ANY 
REPRESENTATIONS, GUARANTEE, OR WARRANTY OF ANY KIND OR IN ANY WAY RELATED TO 
SUPPORT, INDEMNITY, ERROR FREE OR UNINTERRUPTED OPERA TION, OR THAT IT IS FREE 
FROM DEFECTS OR VIRUSES.  ALL OBLIGATIONS ARE HEREBY DISCLAIMED - WHETHER 
EXPRESS, IMPLIED, OR STATUTORY - INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED 
WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, 
ACCURACY, COMPLETENESS, OPERABILITY, QUALITY OF SERVICE, OR NON-INFRINGEMENT. 
IN NO EVENT SHALL ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, PUNITIVE,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, REVENUE, DATA, OR PROFITS; OR 
BUSINESS INTERRUPTION) HOWEVER CAUSED OR BASED ON ANY THEORY OF LIABILITY 
ARISING IN ANY WAY RELATED TO THIS MATERIAL, EVEN IF ADVISED OF THE POSSIBILITY 
OF SUCH DAMAGE. THE ENTIRE AND AGGREGATE LIABILITY OF ADVANCED MICRO DEVICES, 
INC. AND ANY COPYRIGHT HOLDERS AND CONTRIBUTORS SHALL NOT EXCEED TEN DOLLARS 
(US $10.00). ANYONE REDISTRIBUTING OR ACCESSING OR USING THIS MATERIAL ACCEPTS 
THIS ALLOCATION OF RISK AND AGREES TO RELEASE ADVANCED MICRO DEVICES, INC. AND 
ANY COPYRIGHT HOLDERS AND CONTRIBUTORS FROM ANY AND ALL LIABILITIES, 
OBLIGATIONS, CLAIMS, OR DEMANDS IN EXCESS OF TEN DOLLARS (US $10.00). THE 
FOREGOING ARE ESSENTIAL TERMS OF THIS LICENSE AND, IF ANY OF THESE TERMS ARE 
CONSTRUED AS UNENFORCEABLE, FAIL IN ESSENTIAL PURPOSE, OR BECOME VOID OR 
DETRIMENTAL TO ADVANCED MICRO DEVICES, INC. OR ANY COPYRIGHT HOLDERS OR 
CONTRIBUTORS FOR ANY REASON, THEN ALL RIGHTS TO REDISTRIBUTE, ACCESS OR USE 
THIS MATERIAL SHALL TERMINATE IMMEDIATELY. MOREOVER, THE FOREGOING SHALL 
SURVIVE ANY EXPIRATION OR TERMINATION OF THIS LICENSE OR ANY AGREEMENT OR 
ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE IS HEREBY PROVIDED, AND BY REDISTRIBUTING OR ACCESSING OR USING THIS 
MATERIAL SUCH NOTICE IS ACKNOWLEDGED, THAT THIS MATERIAL MAY BE SUBJECT TO 
RESTRICTIONS UNDER THE LAWS AND REGULATIONS OF THE UNITED STATES OR OTHER 
COUNTRIES, WHICH INCLUDE BUT ARE NOT LIMITED TO, U.S. EXPORT CONTROL LAWS SUCH 
AS THE EXPORT ADMINISTRATION REGULATIONS AND NATIONAL SECURITY CONTROLS AS 
DEFINED THEREUNDER, AS WELL AS STATE DEPARTMENT CONTROLS UNDER THE U.S. 
MUNITIONS LIST. THIS MATERIAL MAY NOT BE USED, RELEASED, TRANSFERRED, IMPORTED,
EXPORTED AND/OR RE-EXPORTED IN ANY MANNER PROHIBITED UNDER ANY APPLICABLE LAWS, 
INCLUDING U.S. EXPORT CONTROL LAWS REGARDING SPECIFICALLY DESIGNATED PERSONS, 
COUNTRIES AND NATIONALS OF COUNTRIES SUBJECT TO NATIONAL SECURITY CONTROLS. 
MOREOVER, THE FOREGOING SHALL SURVIVE ANY EXPIRATION OR TERMINATION OF ANY 
LICENSE OR AGREEMENT OR ACCESS OR USE RELATED TO THIS MATERIAL.

NOTICE REGARDING THE U.S. GOVERNMENT AND DOD AGENCIES: This material is 
provided with "RESTRICTED RIGHTS" and/or "LIMITED RIGHTS" as applicable to 
computer software and technical data, respectively. Use, duplication, 
distribution or disclosure by the U.S. Government and/or DOD agencies is 
subject to the full extent of restrictions in all applicable regulations, 
including those found at FAR52.227 and DFARS252.227 et seq. and any successor 
regulations thereof. Use of this material by the U.S. Government and/or DOD 
agencies is acknowledgment of the proprietary rights of any copyright holders 
and contributors, including those of Advanced Micro Devices, Inc., as well as 
the provisions of FAR52.227-14 through 23 regarding privately developed and/or 
commercial computer software.

This license forms the entire agreement regarding the subject matter hereof and 
supersedes all proposals and prior discussions and writings between the parties 
with respect thereto. This license does not affect any ownership, rights, title,
or interest in, or relating to, this material. No terms of this license can be 
modified or waived, and no breach of this license can be excused, unless done 
so in a writing signed by all affected parties. Each term of this license is 
separately enforceable. If any term of this license is determined to be or 
becomes unenforceable or illegal, such term shall be reformed to the minimum 
extent necessary in order for this license to remain in effect in accordance 
with its terms as modified by such reformation. This license shall be governed 
by and construed in accordance with the laws of the State of Texas without 
regard to rules on conflicts of law of any state or jurisdiction or the United 
Nations Convention on the International Sale of Goods. All disputes arising out 
of this license shall be subject to the jurisdiction of the federal and state 
courts in Austin, Texas, and all defenses are hereby waived concerning personal 
jurisdiction and venue of these courts.

============================================================ */
public class RadixSort {
	
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

	final int RADIX = 8;
	final int RADICES = 1 << RADIX;
	
	private Histogram m_histogram = new Histogram();
	private Permute m_permute = new Permute();
	
	public void sort(Device d,  int[] data)
	{
		int groupSize = d.getMaxWorkGroupSize();
		int elementCount = nextPow2(data.length);
		int mulFactor = d.getMaxWorkGroupSize() * RADICES;

	    if(elementCount < mulFactor)
	        elementCount = mulFactor;
	    else
	        elementCount = (elementCount / mulFactor) * mulFactor;
	    
	    m_histogram.unsortedData = new int[elementCount];
	    System.arraycopy(data, 0, m_histogram.unsortedData, 0, data.length);
	    Arrays.fill(m_histogram.unsortedData, data.length +1 , elementCount, Integer.MAX_VALUE);
	    
	    m_permute.unsortedData = m_histogram.unsortedData;
	    
	    m_permute.sortedData = new int[elementCount];
	    Arrays.fill( m_permute.sortedData, 0 , elementCount, 0);
	    
	    int numGroups = elementCount / mulFactor;
	    int tempSize = numGroups * d.getMaxWorkGroupSize() * RADICES;
	    m_histogram.buckets = new int[tempSize];
	    m_histogram.sharedArray = new int[d.getMaxWorkGroupSize() * RADICES];
	    
	    m_permute.scannedBuckets = m_histogram.buckets;
	    
	    m_permute.scannedBuckets = new int[tempSize];
	    m_permute.sharedBuckets = m_histogram.sharedArray;
	    
	    Range r = d.createRange(elementCount / RADICES, groupSize);
	    
	    for(int bits = 0; bits < 4 * RADIX; bits += RADIX)
	    {
	        // Calculate thread-histograms
	    	m_histogram.shiftCount = bits;
	    	
	    	m_histogram.execute(r);
	        

	        // Scan the histogram
	        int sum = 0;
	        for(int i = 0; i < RADICES; ++i)
	        {
	            for(int j = 0; j < numGroups; ++j)
	            {
	                for(int k = 0; k < groupSize; ++k)
	                {
	                    int index = j * groupSize * RADICES + k * RADICES + i;
	                    int value = m_histogram.buckets[index];
	                    m_histogram.buckets[index] = sum;
	                    sum += value;
	                }
	            }
	        }

	        // Permute the element to appropriate place
	        m_permute.execute(r);
	        

	        if (bits +  RADIX < 4 * RADIX)
	        	// Current output now becomes the next input
	        	System.arraycopy(m_permute.sortedData, 0, m_histogram.unsortedData, 0, m_histogram.unsortedData.length);
	        
	    }
	    
	    System.arraycopy(m_permute.sortedData, 0, data, 0, data.length);
	}
	
	class Histogram extends Kernel 
	{
		
		public int[] unsortedData;
		public int[] buckets;
		public int shiftCount;
		public int[] sharedArray;
		

		@Override
		public void run() {
			int localId = getLocalId();
			int globalId = getGlobalId();
			int groupId = getGroupId();
		    int groupSize = getLocalSize();
		    

		    /* Initialize shared array to zero */
		    for(int i = 0; i < RADICES; ++i)
		        sharedArray[localId * RADICES + i] = 0;

		    localBarrier();
		    
		    
		    /* Calculate thread-histograms */
		    for(int i = 0; i < RADICES; ++i)
		    {
		        int value = unsortedData[globalId * RADICES + i];
		        value = (value >> shiftCount) & 0xFF;
		        sharedArray[localId * RADICES + value]++;
		    }
		    
		    localBarrier();
		    
		    /* Copy calculated histogram bin to global memory */
		    for(int i = 0; i < RADICES; ++i)
		    {
		        int bucketPos = groupId * RADICES * groupSize + localId * RADICES + i;
		        buckets[bucketPos] = sharedArray[localId * RADICES + i];
		    }	
		}
		
	}
	
	class Permute extends Kernel 
	{
		public int[] unsortedData;
		public int[] sortedData;
		public int[] scannedBuckets;
		public int shiftCount;
		public int[] sharedBuckets;


		@Override
		public void run() {
			
			int localId = getLocalId();
			int globalId = getGlobalId();
			int groupId = getGroupId();
		    int groupSize = getLocalSize();
		    
			 /* Copy prescaned thread histograms to corresponding thread shared block */
		    for(int i = 0; i < RADICES; ++i)
		    {
		        int bucketPos = groupId * RADICES * groupSize + localId * RADICES + i;
		        sharedBuckets[localId * RADICES + i] = scannedBuckets[bucketPos];
		    }

		    localBarrier();
		    
		    /* Premute elements to appropriate location */
		    for(int i = 0; i < RADICES; ++i)
		    {
		        int value = unsortedData[globalId * RADICES + i];
		        value = (value >> shiftCount) & 0xFF;
		        int index = sharedBuckets[localId * RADICES + value];
		        sortedData[index] = unsortedData[globalId * RADICES + i];
		        sharedBuckets[localId * RADICES + value] = index + 1;
		        localBarrier();

		    }
		}
		
	}
	
}
