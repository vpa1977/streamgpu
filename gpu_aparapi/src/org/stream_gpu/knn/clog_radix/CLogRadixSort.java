package org.stream_gpu.knn.clog_radix;

import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLDevice;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLProgram;

public class CLogRadixSort {
    private int reduceWorkGroupSize;    ///< Work group size for the initial reduce phase
    private int scanWorkGroupSize;      ///< Work group size for the middle scan phase
    private int scatterWorkGroupSize;   ///< Work group size for the final scatter phase
    private int scatterWorkScale;       ///< Elements per work item for the final scan/scatter phase
    private int scatterSlice;           ///< Number of work items that cooperate
    private int scanBlocks;             ///< Maximum number of items in the middle phase
    private int keySize;                ///< Size of the key type
    private int valueSize;              ///< Size of the value type
    private int radix;              ///< Sort radix
    private int radixBits;          ///< Number of bits forming radix
    private CLProgram program;             ///< Program containing the kernels
    private CLKernel reduceKernel;         ///< Initial reduction kernel
    private CLKernel scanKernel;           ///< Middle-phase scan kernel
    private CLKernel scatterKernel;        ///< Final scan/scatter kernel
    private CLBuffer<Integer> histogram;            ///< Histogram of the blocks by radix
    private CLBuffer<Integer> tmpKeys;              ///< User-provided buffer to hold temporary keys
    private CLBuffer<Integer> tmpValues;            ///< User-provided buffer to hold temporary values

    public CLogRadixSort(CLContext context, CLDevice device) 
    {
    	
    }
	
}
